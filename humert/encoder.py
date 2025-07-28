"""
Flash Linear Transformer Encoder with FAVOR+ Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from config.model_config import ModelConfig


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class FAVORPlusAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        # FAVOR+ parameters
        self.num_features = max(64, self.head_dim)
        self.random_features = None
        
        # ALiBi-style bias
        slopes = torch.tensor([1.0 / (2 ** (8 * i / config.num_attention_heads)) 
                              for i in range(config.num_attention_heads)])
        self.register_buffer('alibi_slopes', slopes.view(1, config.num_attention_heads, 1, 1))
        
    def create_random_features(self, batch_size: int, seq_len: int, device: torch.device):
        if self.random_features is None or self.random_features.shape[0] != batch_size:
            self.random_features = torch.randn(
                batch_size, self.num_heads, self.num_features, self.head_dim,
                device=device, dtype=torch.float32
            ) / math.sqrt(self.head_dim)
    
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        # FAVOR+ kernel approximation
        x_proj = torch.einsum('bhld,bhfd->bhlf', x, self.random_features)
        return F.relu(x_proj) / math.sqrt(self.num_features)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Create random features for FAVOR+
        self.create_random_features(batch_size, seq_len, hidden_states.device)
        
        # Apply FAVOR+ approximation
        q_prime = self.phi(q * self.scale)
        k_prime = self.phi(k)
        
        # Linear attention computation O(n)
        kv = torch.einsum('bhlf,bhld->bhfd', k_prime, v)
        out = torch.einsum('bhlf,bhfd->bhld', q_prime, kv)
        
        # Add ALiBi bias (positional encoding)
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).unsqueeze(0)
        alibi_bias = self.alibi_slopes * position_ids
        out = out + alibi_bias
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_dim, config.feedforward_dim)
        self.dense2 = nn.Linear(config.feedforward_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense2(self.dropout(F.gelu(self.dense1(x))))


class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = FAVORPlusAttention(config)
        self.feedforward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN architecture
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states) + residual
        
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.feedforward(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual
        
        return hidden_states


class FlashLinearTransformerEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Adapter gating mechanism
        self.adapter_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.adapter_dim),
                nn.ReLU(),
                nn.Linear(config.adapter_dim, config.hidden_dim),
                nn.Sigmoid()
            ) for _ in range(config.num_layers)
        ])
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask)
            
            # Apply adapter gating
            gate = self.adapter_gates[i](hidden_states)
            hidden_states = hidden_states * gate
            
        return self.final_norm(hidden_states)