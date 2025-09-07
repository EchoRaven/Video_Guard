# --------------------------------------------------------
# InternVL Streaming
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union, Dict, Any
import torch
from torch import nn
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_internvl_streaming import InternVLStreamingConfig
from .modeling_internvl_chat import InternVLChatModel
from .conversation import get_conv_template

logger = logging.get_logger(__name__)


class InternVLStreamingModel(InternVLChatModel):
    """Streaming Video LLM based on InternVL3"""
    
    config_class = InternVLStreamingConfig
    
    def __init__(self, config: InternVLStreamingConfig, **kwargs):
        super().__init__(config, **kwargs)
        
        # Streaming state management
        self.streaming_state = {}
        self.frame_buffer = {}
        
        # Streaming text labels (no special tokens needed)
        self.continue_label = "<continue>"
        self.summary_label = "<summary>"
        self.start_streaming_label = "<start streaming analysis>"
        
    def reset_streaming_state(self, session_id: str):
        """Reset streaming state for a session"""
        if session_id in self.streaming_state:
            del self.streaming_state[session_id]
        if session_id in self.frame_buffer:
            del self.frame_buffer[session_id]
    
    def process_frame_streaming(
        self,
        session_id: str,
        pixel_values: torch.FloatTensor,
        user_prompt: str,
        tokenizer,
        generation_config: Optional[GenerationConfig] = None,
        return_logits: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single frame in streaming mode
        
        Args:
            session_id: Unique session identifier
            pixel_values: Frame pixel values [1, C, H, W]
            user_prompt: User's initial prompt
            tokenizer: Text tokenizer
            generation_config: Generation configuration
            
        Returns:
            Dict containing action type and content
        """
        device = pixel_values.device
        
        # Initialize session if not exists
        if session_id not in self.streaming_state:
            # Add streaming analysis prefix to user prompt
            formatted_prompt = f"{self.start_streaming_label} {user_prompt}"
            self.streaming_state[session_id] = {
                'user_prompt': user_prompt,  # Original prompt
                'formatted_prompt': formatted_prompt,  # With streaming prefix
                'current_shot_frames': [],
                'shot_summaries': [],
                'cumulative_text_context': "",  # 累积的文本context（保留continue/summary信息）
                'total_frames_processed': 0,
                'streaming_sequence': []  # Track the complete streaming sequence
            }
            self.frame_buffer[session_id] = []
        
        # Add frame to current shot
        self.streaming_state[session_id]['current_shot_frames'].append(pixel_values)
        self.frame_buffer[session_id].append(pixel_values)
        self.streaming_state[session_id]['total_frames_processed'] += 1
        
        # Let the model decide whether to continue or generate summary
        # This is done by asking the model to analyze the current frames
        action = self._let_model_decide_action(session_id, tokenizer, generation_config, **kwargs)
        
        if action == 'summary':
            return self._generate_shot_summary(session_id, tokenizer, generation_config, **kwargs)
        else:
            return self._generate_continue_response(session_id, tokenizer, generation_config, **kwargs)
    
    def _let_model_decide_action(
        self,
        session_id: str,
        tokenizer,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """Let the model decide whether to continue or generate summary based on training pattern"""
        state = self.streaming_state[session_id]
        current_frames = state['current_shot_frames']
        
        # Need at least 1 frame to make a decision
        if len(current_frames) < 1:
            return 'continue'
        
        try:
            # 使用和训练时完全一致的格式：真实图像输入 + 简单prompt
            current_frame = current_frames[-1]  # 当前最新一帧
            
            # 确保frame格式正确 [1, 3, H, W]
            if len(current_frame.shape) == 5:
                pixel_values = current_frame.squeeze(1)
            else:
                pixel_values = current_frame
                
            # 确保在正确设备和数据类型
            if pixel_values.device != self.device:
                pixel_values = pixel_values.to(self.device)
            if self.device.type == 'cuda':
                pixel_values = pixel_values.to(torch.bfloat16)
            else:
                pixel_values = pixel_values.to(torch.float32)
            
            # 构建prompt：cumulative context + 当前frame
            # 这样模型可以基于之前的continue/summary决定当前action
            cumulative_context = state.get('cumulative_text_context', '')
            
            if cumulative_context:
                # 有之前的context，包含在prompt中
                context_prompt = cumulative_context + "\n" + "<img><IMG_CONTEXT>" * 256 + "</img>"
            else:
                # 第一帧，没有之前的context
                context_prompt = "<img><IMG_CONTEXT>" * 256 + "</img>"
            
            # 调用模型，期望输出<continue>或<summary>
            response = self.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=context_prompt,
                generation_config={'max_new_tokens': 20, 'do_sample': False}
            )
            
            # 解析模型的输出
            content = str(response).strip().lower()
            
            # 模型应该输出 "<continue>" 或 "<summary> ..."
            if content.startswith('<continue>'):
                return 'continue'
            elif content.startswith('<summary>'):
                return 'summary'
            else:
                # 如果模型输出了实际内容描述，当作summary
                if len(content) > 10:
                    return 'summary'
                else:
                    return 'continue'
                    
        except Exception as e:
            # 如果生成失败，使用简单的启发式
            frame_count = len(current_frames)
            # 前3帧continue，第4帧开始有可能summary
            if frame_count <= 3:
                return 'continue'
            elif frame_count >= 8:
                return 'summary'  # 避免过长
            else:
                # 中等长度时随机决策，模拟训练数据的多样性
                import random
                return 'summary' if random.random() > 0.6 else 'continue'
    
    def _generate_decision_response(
        self,
        prompt: str,
        current_frames: List[torch.FloatTensor],
        tokenizer,
        generation_config: Optional[GenerationConfig] = None
    ) -> str:
        """让模型生成决策响应（continue或summary）"""
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # 转换为chat方法期望的配置格式
        chat_config = {
            'max_new_tokens': 50,  # 决策只需要很短的回复
            'do_sample': False,    # 使用确定性生成获得一致的决策
            'temperature': 0.1,    # 低温度确保决策稳定
        }
        
        # 使用最后一帧进行决策
        if current_frames and len(current_frames) > 0:
            frame = current_frames[-1]  # 使用最新帧
            if len(frame.shape) == 5:  # [1, 1, 3, 448, 448]
                pixel_values = frame.squeeze(1)  # -> [1, 3, 448, 448]
            elif len(frame.shape) == 4:  # [1, 3, 448, 448]
                pixel_values = frame
            else:
                pixel_values = frame.unsqueeze(0)
        else:
            # 如果没有帧，创建dummy frame
            pixel_values = torch.randn(1, 3, 448, 448).to(self.device).to(torch.bfloat16)
        
        # 确保pixel_values在正确的设备和数据类型
        pixel_values = pixel_values.to(self.device).to(torch.bfloat16)
        
        # 使用chat方法生成决策
        response = self.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=chat_config
        )
        
        return response
    
    def _generate_continue_response(
        self,
        session_id: str,
        tokenizer,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate continue response for incomplete shot"""
        state = self.streaming_state[session_id]
        
        # continue时不立即更新cumulative_text_context
        # 等到summary时一次性添加该shot的所有continue
        state['streaming_sequence'].append({
            'action': 'continue',
            'frames': state['current_shot_frames'].copy(),
            'content': self.continue_label
        })
        
        return {
            'action': 'continue',
            'content': self.continue_label,
            'frames_processed': len(state['current_shot_frames']),
            'message': 'Processing continues...'
        }
    
    def _generate_shot_summary(
        self,
        session_id: str,
        tokenizer,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate summary for completed shot"""
        state = self.streaming_state[session_id]
        current_frames = state['current_shot_frames']  # This is a list of tensors
        
        # Create prompt for summary generation
        prompt = self._create_streaming_prompt(
            state['formatted_prompt'],
            current_frames,  # Pass the list of tensors
            tokenizer,
            include_summary=True
        )
        
        # Generate summary response
        summary = self._generate_streaming_response(
            prompt, current_frames, tokenizer, generation_config, **kwargs
        )
        
        # Store shot summary
        shot_info = {
            'frames': state['current_shot_frames'].copy(),
            'summary': summary,
            'frame_count': len(state['current_shot_frames'])
        }
        state['shot_summaries'].append(shot_info)
        
        # 更新cumulative_text_context: 添加该shot的所有continue + summary
        # 与训练时格式保持一致：先所有continue，再summary
        summary_content = f"{self.summary_label} {summary}"
        shot_continues = [self.continue_label] * len(state['current_shot_frames'])
        shot_content = shot_continues + [summary_content]
        
        if state['cumulative_text_context']:
            state['cumulative_text_context'] += "\n" + "\n".join(shot_content)
        else:
            state['cumulative_text_context'] = "\n".join(shot_content)
        
        # Add summary action to streaming sequence
        state['streaming_sequence'].append({
            'action': 'summary',
            'frames': state['current_shot_frames'].copy(),
            'content': summary_content
        })
        
        # Clear current shot frames but keep summary
        state['current_shot_frames'] = []
        
        # Manage KV cache if enabled
        if self.config.enable_kv_cache_management:
            self._manage_kv_cache(session_id, shot_info)
        
        return {
            'action': 'summary',
            'content': summary_content,  # 使用包含<summary>标签的版本
            'frames_processed': shot_info['frame_count'],
            'shot_count': len(state['shot_summaries']),
            'message': f'Shot summary completed: {summary}'
        }
    
    def generate_final_response(
        self,
        session_id: str,
        tokenizer,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate final response for entire video"""
        if session_id not in self.streaming_state:
            return {'error': 'Session not found'}
        
        state = self.streaming_state[session_id]
        
        # Create final response prompt
        final_prompt = self._create_final_response_prompt(
            state['formatted_prompt'],
            state['shot_summaries'],
            tokenizer
        )
        
        # Generate final response
        final_response = self._generate_final_response(
            final_prompt, state['shot_summaries'], tokenizer, generation_config, **kwargs
        )
        
        # Add final response to streaming sequence
        state['streaming_sequence'].append({
            'action': 'final_response',
            'frames': [],  # No specific frames for final response
            'content': final_response
        })
        
        return {
            'action': 'final_response',
            'content': final_response,
            'total_frames': state['total_frames_processed'],
            'total_shots': len(state['shot_summaries']),
            'message': f'Video analysis completed: {final_response}'
        }
    
    def _create_decision_prompt(
        self,
        user_prompt: str,
        current_frames: List[torch.FloatTensor],
        tokenizer
    ) -> str:
        """Create prompt for decision making"""
        template = get_conv_template(self.template)
        template.system_message = self.system_message
        
        # Add user prompt
        template.append_message(template.roles[0], user_prompt)
        
        # Add image tokens - use <image> marker and let InternVL3 handle the replacement
        # InternVL3 will automatically replace <image> with the correct image tokens
        image_tokens = "<image>"
        
        # Ask model to decide if shot is complete
        template.append_message(template.roles[1], f"{image_tokens}\nPlease analyze these video frames and determine if the current shot is complete. If the shot is complete, please indicate so. If more frames are needed, please continue.")
        
        return template.get_prompt()
    
    def _create_streaming_prompt(
        self,
        user_prompt: str,
        current_frames: List[torch.FloatTensor],
        tokenizer,
        include_summary: bool = False
    ) -> str:
        """Create prompt for streaming generation"""
        template = get_conv_template(self.template)
        template.system_message = self.system_message
        
        # Add user prompt
        template.append_message(template.roles[0], user_prompt)
        
        # Add image tokens - use <image> marker and let InternVL3 handle the replacement
        # InternVL3 will automatically replace <image> with the correct image tokens
        image_tokens = "<image>"
        
        if include_summary:
            # Request summary
            template.append_message(template.roles[1], f"{image_tokens}\nPlease generate a summary for this video shot.")
        else:
            # Request continue analysis
            template.append_message(template.roles[1], f"{image_tokens}\nPlease analyze this video frame. If the shot is not complete, please continue.")
        
        return template.get_prompt()
    
    def _create_final_response_prompt(
        self,
        user_prompt: str,
        shot_summaries: List[Dict],
        tokenizer
    ) -> str:
        """Create prompt for final video response - 使用完整的streaming历史"""
        # 使用cumulative_text_context，包含完整的continue/summary历史
        # 这样final response就基于完整的streaming过程
        session_id = None
        for sid, state in self.streaming_state.items():
            if len(state['shot_summaries']) == len(shot_summaries):
                session_id = sid
                break
        
        if session_id and 'cumulative_text_context' in self.streaming_state[session_id]:
            # 使用完整的streaming历史作为输入
            cumulative_context = self.streaming_state[session_id]['cumulative_text_context']
            final_input_prompt = f"{cumulative_context}<|im_end|>"
        else:
            # 降级方案：只使用summaries（保持向后兼容）
            summaries_only = []
            for shot in shot_summaries:
                summaries_only.append(shot['summary'])
            summaries_text = "\n".join(summaries_only)
            final_input_prompt = f"{summaries_text}<|im_end|>"
        
        return final_input_prompt
    
    def _generate_decision_response(
        self,
        prompt: str,
        current_frames: List[torch.FloatTensor],
        tokenizer,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """Generate response for decision prompt using InternVL3's chat method"""
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Convert generation_config to dict format expected by chat method
        chat_config = {
            'max_new_tokens': 256,  # Fixed value to avoid issues
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'repetition_penalty': 1.1
        }
        
        # Convert current_frames list to proper format for InternVL3
        # Each frame should be [1, num_patches, C, H, W] format
        if current_frames and len(current_frames) > 0:
            # Take the first frame as reference for decision
            frame = current_frames[0]  # [C, H, W]
            # Add batch dimension and ensure proper format
            if len(frame.shape) == 3:
                pixel_values = frame.unsqueeze(0)  # [1, C, H, W]
            else:
                pixel_values = frame
        else:
            # Create dummy frame if no frames available
            pixel_values = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Use the chat method - MUST work, no fallback
        if tokenizer is None:
            raise ValueError("Tokenizer cannot be None for streaming model")
        
        response = self.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=chat_config
        )
        return response
    
    def _generate_streaming_response(
        self,
        prompt: str,
        current_frames: List[torch.FloatTensor],
        tokenizer,
        generation_config: Optional[GenerationConfig] = None,
        return_logits: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate streaming response using InternVL3's chat method"""
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Convert generation_config to dict format expected by chat method
        chat_config = {
            'max_new_tokens': 256,  # Fixed value to avoid issues
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.9,
            'repetition_penalty': 1.1
        }
        
        # Convert current_frames list to proper format for InternVL3
        # Each frame should be [1, C, H, W] format
        if current_frames and len(current_frames) > 0:
            # Take the first frame as reference for response
            frame = current_frames[0]  # Should be [1, 1, 3, 448, 448]
            # Convert to proper format [1, 3, 448, 448]
            if len(frame.shape) == 5:  # [1, 1, 3, 448, 448]
                pixel_values = frame.squeeze(1)  # Remove the second dimension -> [1, 3, 448, 448]
            elif len(frame.shape) == 4:  # [1, 3, 448, 448]
                pixel_values = frame
            elif len(frame.shape) == 3:  # [3, 448, 448]
                pixel_values = frame.unsqueeze(0)  # [1, 3, 448, 448]
            else:
                pixel_values = frame
            
            # Ensure correct data type (convert to bfloat16 if model is on GPU)
            if pixel_values.device.type == 'cuda':
                pixel_values = pixel_values.to(torch.bfloat16)
            else:
                pixel_values = pixel_values.to(torch.float32)
        else:
            # Create dummy frame if no frames available
            pixel_values = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Use the chat method - MUST work, no fallback
        if tokenizer is None:
            raise ValueError("Tokenizer cannot be None for streaming model")
        
        response = self.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=chat_config
        )
        return response
    
    def _generate_final_response(
        self,
        prompt: str,
        shot_summaries: List[Dict],
        tokenizer,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> str:
        """Generate final response - 让模型自己学会生成<response>标签"""
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # 使用传入的prompt（已经包含完整的streaming历史）
        final_input_prompt = prompt
        
        # 使用language model生成，期望它补全<response>后的内容
        inputs = tokenizer(final_input_prompt, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.language_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (skip the input)
        input_length = inputs['input_ids'].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return response.strip()
    
    def _manage_kv_cache(self, session_id: str, shot_info: Dict[str, Any]):
        """简化的KV cache管理：只清理frame数据，不管理KV cache"""
        # 不进行复杂的KV cache操作，让模型自然管理
        # 只确保frame数据被清理（已在调用方完成）
        logger.debug(f"🧹 Shot {len(self.streaming_state[session_id]['shot_summaries'])} completed. "
                    f"Frame data cleared, KV cache preserved.")
    

    
    def get_streaming_status(self, session_id: str) -> Dict[str, Any]:
        """Get current streaming status for a session"""
        if session_id not in self.streaming_state:
            return {'status': 'not_started'}
        
        state = self.streaming_state[session_id]
        return {
            'status': 'active',
            'total_frames_processed': state['total_frames_processed'],
            'current_shot_frames': len(state['current_shot_frames']),
            'shots_completed': len(state['shot_summaries']),
            'user_prompt': state['user_prompt'],
            'streaming_sequence': state['streaming_sequence']
        }
    
    def get_complete_streaming_sequence(self, session_id: str) -> Dict[str, Any]:
        """Get complete streaming sequence for training"""
        if session_id not in self.streaming_state:
            return {'error': 'Session not found'}
        
        state = self.streaming_state[session_id]
        
        # Build complete sequence string
        sequence_parts = []
        sequence_parts.append(state['formatted_prompt'])  # Use formatted prompt with streaming prefix
        
        for item in state['streaming_sequence']:
            if item['action'] == 'continue':
                sequence_parts.append(self.continue_label)
            elif item['action'] == 'summary':
                sequence_parts.append(item['content'])  # Already contains summary label
            elif item['action'] == 'final_response':
                sequence_parts.append(item['content'])
        
        complete_sequence = ' '.join(sequence_parts)
        
        return {
            'complete_sequence': complete_sequence,
            'streaming_sequence': state['streaming_sequence'],
            'shot_summaries': state['shot_summaries'],
            'total_frames': state['total_frames_processed'],
            'total_shots': len(state['shot_summaries'])
        }
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        """
        Forward pass for streaming model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            pixel_values: Image pixel values (optional for text-only training)
            labels: Training labels
            return_dict: Whether to return dict
        """
        # For text-only training (no pixel_values), use language model directly
        if pixel_values is None:
            # Text-only mode for training
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=return_dict,
                **kwargs
            )
            return outputs
        
        # For inference with images, use the full streaming logic
        # This would be the normal streaming behavior
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            return_dict=return_dict,
            **kwargs
        )

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings"""
        return self.language_model.set_output_embeddings(new_embeddings)
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None) -> torch.nn.Embedding:
        """Resize token embeddings"""
        return self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
