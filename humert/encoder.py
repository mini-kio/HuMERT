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

        self.num_features = max(64, self.head_dim)
        random_projection = torch.randn(self.num_heads, self.head_dim, self.num_features) / math.sqrt(self.head_dim)
        self.register_buffer('random_projection', random_projection)

        slopes = torch.tensor([1.0 / (2 ** (8 * i / config.num_attention_heads)) for i in range(config.num_attention_heads)])
        self.register_buffer('alibi_slopes', slopes.view(1, config.num_attention_heads, 1, 1))
        
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """FAVOR+ positive random features mapping (ReLU kernel approx)."""
        # x: [B, H, L, D]; projection: [H, D, F]
        x_proj = torch.einsum('bhld,hdf->bhlf', x, self.random_projection)
        # Stabilize: subtract max per token to reduce overflow before exp if used.
        # Here we keep ReLU feature map; add epsilon for numerical stability.
        return F.relu(x_proj) + 1e-6
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply FAVOR+ approximation
        q_prime = self.phi(q * self.scale)  # [B,H,L,F]
        k_prime = self.phi(k)               # [B,H,L,F]

        # Optional causal / padding mask support
        if attention_mask is not None:
            # attention_mask: [B, L] (1 = keep, 0 = mask)
            mask = attention_mask[:, None, :, None].to(q_prime.dtype)
            k_prime = k_prime * mask
            v = v * mask

        # Compute denominator for normalization: q' * sum_k k'
        k_sum = k_prime.sum(dim=2)  # [B,H,F]
        denom = torch.einsum('bhlf,bhf->bhl', q_prime, k_sum) + 1e-6  # [B,H,L]

        # Compute numerator: (q' * (k'^T v)) using associative trick
        kv = torch.einsum('bhlf,bhld->bhfd', k_prime, v)  # [B,H,F,D]
        out = torch.einsum('bhlf,bhfd->bhld', q_prime, kv)  # [B,H,L,D]
        out = out / denom[..., None]  # Normalize
        
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
        use_ckpt = getattr(self.config, 'use_activation_checkpointing', False) and self.training
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint
        for i, layer in enumerate(self.layers):
            if use_ckpt:
                hidden_states = checkpoint(lambda hs, am=None: layer(hs, am), hidden_states, attention_mask)
            else:
                hidden_states = layer(hidden_states, attention_mask)
            gate = self.adapter_gates[i](hidden_states)
            hidden_states = hidden_states * gate
        return self.final_norm(hidden_states)