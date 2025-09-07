# Video-Guard Streaming Model Package

## 包含内容

### streaming_model/
- `modeling_internvl_streaming.py` - 流式VLM模型主类
- `modeling_internvl_chat.py` - 基础聊天模型 (支持pixel_values=None优化)
- `modeling_intern_vit.py` - 视觉编码器
- `conversation.py` - 对话模板管理
- `configuration_*.py` - 模型配置文件
- `__init__.py` - 模块初始化

### training_testing/  
- `streaming_trainer.py` - 统一训练器 (支持混合batch训练)
- `streaming_dataset.py` - 数据集处理 (支持多帧逐步训练)
- `__init__.py` - 模块初始化

## 主要特性

### 🎯 流式视频安全检测
- 逐帧分析视频内容
- 实时安全检测和决策
- 支持6种安全类别 (C1-C6)

### 🚀 优化的训练流程
- 混合batch训练 (clip + final_response)
- 智能padding和序列长度管理
- 支持pixel_values=None优化
- 多帧逐步学习策略

### ⚡ 技术优化
- 最大序列长度: 4096 tokens
- 批处理大小: 4 (配合梯度累积)
- 支持多标签输出
- 内存高效的视觉处理

## 使用示例

### 训练
```python
from training_testing.streaming_trainer import StreamingVLMTrainer

config = {
    "model_name": "OpenGVLab/InternVL3-1B", 
    "batch_size": 4,
    "max_length": 4096,
    "dataset_file": "your_dataset.jsonl"
}

trainer = StreamingVLMTrainer(config)
trainer.train()
```

### 推理
```python
from streaming_model.modeling_internvl_streaming import InternVLStreamingModel

model = InternVLStreamingModel.from_pretrained("your_model_path")
# 逐帧流式分析...
```

## 数据格式

支持统一的JSONL格式，包含SafeWatch和Shot2Story数据。

## 更新记录

- ✅ 支持多标签unsafe输出
- ✅ 优化pixel_values=None处理  
- ✅ 统一clip和final_response训练
- ✅ 提升序列长度到4096
- ✅ 智能batch排序减少padding
