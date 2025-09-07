# --------------------------------------------------------
# InternVL Streaming
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy
from transformers.utils import logging

from .configuration_internvl_chat import InternVLChatConfig

logger = logging.get_logger(__name__)


class InternVLStreamingConfig(InternVLChatConfig):
    model_type = 'internvl_streaming'
    
    def __init__(
            self,
            frame_sampling_rate=1.0,  # frames per second
            enable_kv_cache_management=True,  # enable KV cache management
            **kwargs):
        # Set default template if not provided
        if 'template' not in kwargs:
            kwargs['template'] = 'internvl2_5'
        super().__init__(**kwargs)
        
        self.frame_sampling_rate = frame_sampling_rate
        self.enable_kv_cache_management = enable_kv_cache_management
        
        logger.info(f'frame_sampling_rate: {self.frame_sampling_rate}')
        logger.info(f'enable_kv_cache_management: {self.enable_kv_cache_management}')

    def to_dict(self):
        output = super().to_dict()
        output['frame_sampling_rate'] = self.frame_sampling_rate
        output['enable_kv_cache_management'] = self.enable_kv_cache_management
        output['model_type'] = self.__class__.model_type
        return output
