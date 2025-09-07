#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom LoRA Trainer for InternVL3-8B
Avoids HuggingFace Trainer issues with LoRA multi-GPU training
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    HfArgumentParser,
    get_linear_schedule_with_warmup
)

# PEFT imports for LoRA
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Import our custom dataset
from Dataloader import StreamingDataset
from train_multi_gpu import DataCollatorForStreaming

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""
    
    model_name_or_path: str = field(
        default="OpenGVLab/InternVL3-8B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )


@dataclass 
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    
    dataset_dir: str = field(
        default="/scratch/czr/Video-Guard/datasets",
        metadata={"help": "Path to the dataset directory"}
    )
    max_samples: List[int] = field(
        default_factory=lambda: [1000, 1000],  # [shot2story, safewatch]
        metadata={"help": "Maximum number of samples to load from each dataset"}
    )
    max_length: int = field(
        default=16384,  # 减少到16K以适应LoRA训练
        metadata={"help": "Maximum sequence length for tokenization"}
    )
    input_size: int = field(
        default=448,
        metadata={"help": "Input image size"}
    )
    max_num_patches: int = field(
        default=12,
        metadata={"help": "Maximum number of patches per image"}
    )


@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration"""
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient training"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA attention dimension (rank)"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA scaling parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        metadata={"help": "Target modules for LoRA adaptation"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"}
    )


@dataclass
class TrainingArguments:
    """Arguments for training configuration"""
    output_dir: str = field(
        default="./output_custom_lora",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Total number of training epochs to perform."}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/CPU for training."}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    learning_rate: float = field(
        default=5e-4,  # 提高学习率，适合LoRA训练
        metadata={"help": "The initial learning rate for AdamW."}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps."}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps."}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints."}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bf16 (mixed) precision instead of 32-bit."}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing to save memory at the expense of slower backward pass."}
    )
    dataloader_num_workers: int = field(
        default=8,
        metadata={"help": "Number of subprocesses to use for data loading."}
    )
    run_name: str = field(
        default="custom-lora-training",
        metadata={"help": "An optional descriptor for the run."}
    )


class CustomLoRATrainer:
    """Custom trainer for LoRA fine-tuning with multi-GPU support"""
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        data_collator,
        training_args: TrainingArguments,
        optimizer=None,
        lr_scheduler=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.args = training_args
        
        # Setup distributed training
        self.is_distributed = dist.is_initialized()
        if self.is_distributed:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else:
            self.local_rank = 0
            self.world_size = 1
            self.global_rank = 0
            
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        # Setup model
        self.model = self.model.to(self.device)
        if self.is_distributed:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
            
        # Setup data loader
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True
            )
        else:
            train_sampler = None
            
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=training_args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Setup optimizer and scheduler
        if optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": training_args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() 
                              if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters, 
                lr=training_args.learning_rate,
                eps=1e-8
            )
        else:
            self.optimizer = optimizer
            
        # Calculate total training steps
        total_steps = len(self.train_dataloader) * training_args.num_train_epochs // training_args.gradient_accumulation_steps
        warmup_steps = int(total_steps * training_args.warmup_ratio)
        
        if lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.lr_scheduler = lr_scheduler
            
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Create output directory
        os.makedirs(training_args.output_dir, exist_ok=True)
        
    def train(self):
        """Main training loop"""
        if self.global_rank == 0:
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(self.train_dataset)}")
            logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {len(self.train_dataloader) * self.args.num_train_epochs // self.args.gradient_accumulation_steps}")
        
        self.model.train()
        
        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            if self.is_distributed:
                self.train_dataloader.sampler.set_epoch(epoch)
                
            epoch_loss = 0.0
            epoch_steps = 0
            
            for step, batch in enumerate(self.train_dataloader):
                loss = self.training_step(batch)
                epoch_loss += loss
                epoch_steps += 1
                
                # Gradient accumulation
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.args.logging_steps == 0 and self.global_rank == 0:
                        avg_loss = epoch_loss / epoch_steps
                        lr = self.lr_scheduler.get_last_lr()[0]
                        logger.info(f"Epoch {epoch+1}/{self.args.num_train_epochs} | "
                                  f"Step {self.global_step} | "
                                  f"Loss: {avg_loss:.4f} | "
                                  f"LR: {lr:.2e}")
                    
                    # Save checkpoint
                    if self.global_step % self.args.save_steps == 0 and self.global_rank == 0:
                        self.save_checkpoint()
                        
            if self.global_rank == 0:
                avg_epoch_loss = epoch_loss / epoch_steps
                logger.info(f"Epoch {epoch+1} completed | Average Loss: {avg_epoch_loss:.4f}")
                
        # Save final checkpoint
        if self.global_rank == 0:
            self.save_checkpoint(final=True)
            logger.info("Training completed!")
            
    def training_step(self, batch):
        """Single training step"""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        pixel_values = batch.get('pixel_values')
        
        if pixel_values is not None and pixel_values.numel() > 0:
            pixel_values = pixel_values.to(self.device)
        else:
            pixel_values = None
            
        # Forward pass - explicitly call with individual parameters
        if pixel_values is not None:
            # 处理多帧视频数据 - 按照MJ-Video方式
            if len(pixel_values.shape) == 4:  # [batch_size, patches, C, H, W]
                # 已经是展平的格式，直接使用
                pixel_values_flat = pixel_values.view(-1, pixel_values.shape[-3], pixel_values.shape[-2], pixel_values.shape[-1])
            elif len(pixel_values.shape) == 5:  # [batch_size, patches, C, H, W]
                # 这是我们的格式：[batch_size, patches, C, H, W]
                pixel_values_flat = pixel_values.view(-1, pixel_values.shape[-3], pixel_values.shape[-2], pixel_values.shape[-1])
            else:
                # Fallback
                pixel_values_flat = pixel_values
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values_flat,
                labels=labels,
                return_dict=True
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
        loss = outputs.loss
        
        # 检查loss是否为NaN或Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Loss is {loss.item()}, skipping this batch")
            return 0.0
        
        # Scale loss for gradient accumulation
        loss = loss / self.args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
        return loss.item()
        
    def save_checkpoint(self, final=False):
        """Save model checkpoint"""
        if final:
            output_dir = self.args.output_dir
        else:
            output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if self.is_distributed:
            model_to_save = self.model.module
        else:
            model_to_save = self.model
            
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training state
        training_state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }
        torch.save(training_state, os.path.join(output_dir, 'training_state.pt'))
        
        logger.info(f"Checkpoint saved to {output_dir}")


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(
            backend='nccl',
            rank=rank,
            world_size=world_size
        )
        
        torch.cuda.set_device(local_rank)
        return True
    return False


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup distributed training
    is_distributed = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    
    # Setup logging
    if global_rank == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        
    logger.info(f"Global rank: {global_rank}, Local rank: {local_rank}")
    
    # Load tokenizer and model
    logger.info(f"Loading model and tokenizer from {model_args.model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=False
    )
    
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        device_map=None  # We'll handle device placement manually
    )
    
    # Ensure all parameters are in the same dtype for bf16 training
    if training_args.bf16:
        # Convert the entire model to bfloat16, including vision model
        model = model.to(torch.bfloat16)
        
        # Specifically ensure vision model components are in bfloat16
        if hasattr(model, 'vision_model'):
            model.vision_model = model.vision_model.to(torch.bfloat16)
        
        # Ensure all parameters and buffers are in bfloat16
        for name, param in model.named_parameters():
            if param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)
        
        for name, buffer in model.named_buffers():
            if buffer.dtype != torch.bfloat16:
                buffer.data = buffer.data.to(torch.bfloat16)
    
    # Initialize special tokens and img_context_token_id
    if not hasattr(tokenizer, 'img_context_token_id') or tokenizer.img_context_token_id is None:
        tokenizer.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        
    if not hasattr(model, 'img_context_token_id') or model.img_context_token_id is None:
        model.img_context_token_id = tokenizer.img_context_token_id
        
    if global_rank == 0:
        logger.info(f"img_context_token_id: {model.img_context_token_id}")
        
        # 确保模型使用与Dataloader相同的max_dynamic_patch配置
        if hasattr(model.config, 'max_dynamic_patch'):
            logger.info(f"Original max_dynamic_patch: {model.config.max_dynamic_patch}")
            model.config.max_dynamic_patch = data_args.max_num_patches  # 使用与Dataloader相同的配置
            logger.info(f"Updated max_dynamic_patch: {model.config.max_dynamic_patch}")
        
        # 同时更新vision_config中的配置
        if hasattr(model.config, 'vision_config') and hasattr(model.config.vision_config, 'max_dynamic_patch'):
            logger.info(f"Original vision_config.max_dynamic_patch: {model.config.vision_config.max_dynamic_patch}")
            model.config.vision_config.max_dynamic_patch = data_args.max_num_patches
            logger.info(f"Updated vision_config.max_dynamic_patch: {model.config.vision_config.max_dynamic_patch}")
    
    # Apply LoRA if enabled
    if lora_args.use_lora:
        if global_rank == 0:
            logger.info("Applying LoRA configuration...")
        
        # Prepare model for LoRA training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules,
            bias=lora_args.lora_bias,
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Ensure all parameters are still in bfloat16 after LoRA application
        if training_args.bf16:
            model = model.to(torch.bfloat16)
            # Force all parameters and buffers to bfloat16
            for name, param in model.named_parameters():
                if param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16)
            for name, buffer in model.named_buffers():
                if buffer.dtype != torch.bfloat16:
                    buffer.data = buffer.data.to(torch.bfloat16)
        
        # Print trainable parameters
        if global_rank == 0:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in model.parameters())
            logger.info(f"LoRA enabled - Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
        
        # Enable gradient checkpointing for LoRA
        model.enable_input_require_grads()
        
        # 确保LoRA参数需要梯度
        lora_params_count = 0
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                lora_params_count += 1
        
        if global_rank == 0:
            logger.info(f"Set requires_grad=True for {lora_params_count} LoRA parameters")
            
        # 验证LoRA参数设置
        trainable_lora_params = sum(p.numel() for name, p in model.named_parameters() 
                                  if 'lora_' in name and p.requires_grad)
        if global_rank == 0:
            logger.info(f"LoRA trainable parameters: {trainable_lora_params:,}")
    else:
        if global_rank == 0:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Full fine-tuning - Trainable parameters: {trainable_params:,}")
    
    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Load dataset
    if global_rank == 0:
        logger.info("Loading training dataset...")
    
    train_dataset = StreamingDataset(
        dataset_file=data_args.dataset_dir,
        tokenizer=tokenizer,
        max_samples=data_args.max_samples,
        max_length=data_args.max_length,
        input_size=data_args.input_size,
        max_num_patches=data_args.max_num_patches
    )
    
    if global_rank == 0:
        logger.info(f"Training dataset loaded: {len(train_dataset)} samples")
    
    # Create data collator with dynamic image token replacement
    data_collator = DataCollatorForStreaming(
        tokenizer=tokenizer,
        max_length=data_args.max_length
    )
    
    # Create trainer
    trainer = CustomLoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
        training_args=training_args
    )
    
    # Start training
    trainer.train()
    
    # Cleanup
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
