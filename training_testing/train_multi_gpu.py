import torch
from typing import Dict, List, Any, Optional
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

class DataCollatorForStreaming:
    """
    Data collator for streaming video data with dynamic image token replacement.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 16384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Get IMG_CONTEXT token ID
        self.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        if self.img_context_token_id == tokenizer.unk_token_id:
            raise ValueError("IMG_CONTEXT token not found in tokenizer vocabulary")
        
        logger.info(f"img_context_token_id: {self.img_context_token_id}")
    
    def _prepare_chat_input_mj_style(self, text: str, num_patches_list: List[int]) -> tuple:
        """
        Prepare chat input in MJ-Video style with dynamic image token replacement.
        
        Args:
            text: Input text with <image> placeholders
            num_patches_list: List of number of patches for each image
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        
        # 计算每个patch的token数量 (InternVL3-8B: 256)
        num_image_token = 256 # 每个patch的token数量
        
        query = text
        # 根据实际的patches数量来插入token
        num_image_placeholders = text.count('<image>')
        
        # 如果没有图像占位符，直接返回原始文本
        if num_image_placeholders == 0:
            model_inputs = self.tokenizer(
                query, 
                return_tensors='pt', 
                truncation=True, 
                max_length=self.max_length
            )
            input_ids = model_inputs['input_ids'].squeeze(0)
            attention_mask = model_inputs['attention_mask'].squeeze(0)
            return input_ids, attention_mask
        
        # 确保patches数量与占位符数量匹配
        if len(num_patches_list) != num_image_placeholders:
            # 如果数量不匹配，使用第一个patches数量或默认值
            if num_patches_list:
                default_patches = num_patches_list[0]
            else:
                default_patches = 1
            num_patches_list = [default_patches] * num_image_placeholders
        
        for i, num_patches in enumerate(num_patches_list):
            # 每个patch需要256个token
            total_image_tokens = num_image_token * num_patches
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * total_image_tokens + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        model_inputs = self.tokenizer(
            query, 
            return_tensors='pt', 
            truncation=True, 
            max_length=self.max_length
        )
        
        input_ids = model_inputs['input_ids'].squeeze(0)
        attention_mask = model_inputs['attention_mask'].squeeze(0)
        
        return input_ids, attention_mask
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching streaming data.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Batched tensors
        """
        # Extract data from features
        texts = [f['text'] for f in features]
        images = [f['pixel_values'] for f in features]
        num_patches_list = [f['num_patches_list'] for f in features]
        
        # Process each sample
        input_ids_list = []
        attention_mask_list = []
        pixel_values_list = []
        
        for text, image, num_patches in zip(texts, images, num_patches_list):
            # Prepare text input with image tokens
            # num_patches is already a list, so we pass it directly
            input_ids, attention_mask = self._prepare_chat_input_mj_style(text, num_patches)
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            
            # Handle image data - 按照MJ-Video方式处理
            if image is not None:
                if len(image.shape) == 4:  # [total_patches, C, H, W] - from Dataloader
                    # 直接使用Dataloader的输出，保持[total_patches, C, H, W]格式
                    pixel_values = image
                elif len(image.shape) == 5:  # [batch_size, num_patches, C, H, W]
                    # 已经是正确的格式，直接使用
                    pixel_values = image.squeeze(0)  # [num_patches, C, H, W]
                else:
                    # Fallback: create placeholder
                    pixel_values = torch.zeros(1, 3, 448, 448)
                
                # Convert to bfloat16 to match model dtype
                pixel_values = pixel_values.to(torch.bfloat16)
                pixel_values_list.append(pixel_values)
            else:
                # 检查文本中是否有<image>占位符
                num_image_placeholders = text.count('<image>')
                if num_image_placeholders > 0:
                    # 有图像占位符但没有图像数据，创建placeholder
                    logger.warning(f"Sample {len(pixel_values_list)} has <image> placeholders but no image data, creating placeholder")
                    pixel_values = torch.zeros(num_image_placeholders, 3, 448, 448, dtype=torch.bfloat16)
                    pixel_values_list.append(pixel_values)
                else:
                    # 没有图像占位符，不添加图像数据
                    pixel_values_list.append(None)
        
        # Pad sequences
        max_length = max(len(ids) for ids in input_ids_list)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            pad_length = max_length - len(input_ids)
            if pad_length > 0:
                input_ids = torch.cat([input_ids, torch.full((pad_length,), self.tokenizer.pad_token_id)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_length)])
            
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
        
        # Create labels for causal LM training
        labels = []
        for i, input_ids in enumerate(padded_input_ids):
            # 创建labels，初始化为-100（忽略loss）
            label = torch.full_like(input_ids, -100)
            
            # 获取对应的文本内容来判断类型
            text = texts[i] if i < len(texts) else ""
            
            if 'final_response' in text or '<|vision_end|>' in text:
                # final_response类型：从<|vision_end|>开始计算loss
                # 1. 先建立labels全都是和input_ids一样，直接克隆
                label = input_ids.clone()
                
                # 2. 找到<|vision_end|>的位置，前面的都变成-100
                # <|vision_end|>是special token，只有一个token ID: 151653
                vision_end_token_id = 151653
                
                # 查找<|vision_end|>的位置
                vision_end_pos = -1
                for i in range(len(input_ids)):
                    if input_ids[i] == vision_end_token_id:
                        vision_end_pos = i
                        break
                
                if vision_end_pos >= 0:
                    # 从<|vision_end|>之后开始计算loss
                    start_pos = vision_end_pos + 1
                    if start_pos < len(input_ids):
                        # 前面的都变成-100
                        label[:start_pos] = -100
                        # 剩下的都算loss（已经是input_ids.clone()）
                    else:
                        # <|vision_end|>是最后一个token，没有需要计算loss的内容
                        # 整个序列都标记为IGNORE
                        label[:] = -100
                else:
                    # 如果没有<|vision_end|>，从后1/3开始计算loss
                    start_pos = len(input_ids) // 3
                    label[:start_pos] = -100
            else:
                # clip类型：跳过user prompt和image tokens，剩下的都算loss
                # 1. 先建立labels全都是和input_ids一样，直接克隆
                label = input_ids.clone()
                
                # 2. 找到第一个<img>的位置，该位置之前的label全换成-100
                img_token_id = 151665  # <img> token ID
                img_positions = []
                for i in range(len(input_ids)):
                    if input_ids[i] == img_token_id:
                        img_positions.append(i)
                
                if len(img_positions) > 0:
                    first_img_pos = img_positions[0]
                    # 第一个<img>之前的所有tokens都标记为IGNORE
                    label[:first_img_pos] = -100
                    
                    # 3. 找到所有<img>, </img>, <IMG_CONTEXT>的位置，变成-100
                    end_img_token_id = 151666  # </img> token ID
                    img_context_token_id = 151667  # <IMG_CONTEXT> token ID
                    
                    for i in range(len(input_ids)):
                        if input_ids[i] in [img_token_id, end_img_token_id, img_context_token_id]:
                            label[i] = -100
                else:
                    # 如果没有<img>，从后1/3开始计算loss
                    start_pos = len(input_ids) // 3
                    label[:start_pos] = -100
            
            labels.append(label)
        
        # Stack tensors - 按照MJ-Video方式
        # 直接在第一个维度上拼接所有pixel_values
        all_pixel_values = []
        for pv in pixel_values_list:
            if pv is not None and pv.shape[0] > 0:
                all_pixel_values.append(pv)
        
        if all_pixel_values:
            # 拼接所有pixel_values
            pixel_values = torch.cat(all_pixel_values, dim=0)  # [total_patches, C, H, W]
        else:
            # 没有图像数据，直接返回None，让模型处理
            pixel_values = None
        
        batch = {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask),
            'pixel_values': pixel_values,  # [total_patches, C, H, W]
            'labels': torch.stack(labels)
        }
        
        # Debug: print shapes
        if batch['pixel_values'] is not None:
            print(f"Debug - pixel_values shape: {batch['pixel_values'].shape}")
        else:
            print("Debug - pixel_values: None")
        print(f"Debug - input_ids shape: {batch['input_ids'].shape}")
        
        return batch
