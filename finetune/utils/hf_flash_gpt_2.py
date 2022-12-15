# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modified HF GPT2 w/flash attention"""

import os
from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from torch import nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2MLP,
    CausalLMOutputWithCrossAttentions,
    GPT2Attention,
    GPT2Block,
    GPT2LMHeadModel,
    GPT2Model,
    GPT2PreTrainedModel,
)


class GPT2FlashAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config=config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.attn_pdrop = config.attn_pdrop

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # rearrange to flash attention form
        key = rearrange(key, 'b h s d -> b s h d')
        value = rearrange(value, 'b h s d -> b s h d')
        query = rearrange(query, 'b h s d -> b s h d')

        # stack
        qkv = torch.stack([query,key,value], dim=2)
        assert qkv.dtype in [torch.float16, torch.bfloat16]

        # flash attention logic
        batch_size = qkv.shape[0]
        seqlen = qkv.shape[1]
        dk = qkv.shape[4]
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        max_s = seqlen
        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=qkv.device)
        attn_pdrop = self.attn_pdrop if self.training else 0.0
        softmax_scale = (1.0 / (dk ** 0.5)) if self.scale_attn_weights else 1.0
        softmax_scale = (softmax_scale / float(self.layer_idx + 1)) if self.scale_attn_by_inverse_layer_idx else softmax_scale
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_seqlens, max_s, attn_pdrop,
            softmax_scale=softmax_scale, causal=True
        )
        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        output = rearrange(output, 'b s h d -> b h s d')

        return output, None


class GPT2FlashBlock(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super(GPT2Block, self).__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2FlashAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2FlashAttention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)


class GPT2FlashModel(GPT2Model):
    def __init__(self, config):
        super(GPT2Model, self).__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2FlashBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class GPT2FlashLMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__(config)

        self.transformer = GPT2FlashModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
