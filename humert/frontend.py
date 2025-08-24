"""
ConvFrontend24k: Raw waveform preprocessing for 24kHz audio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from config.model_config import ModelConfig


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )
        self.norm = nn.GroupNorm(min(32, out_channels // 4), out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvFrontend24k(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Initial conv layer
        self.initial_conv = nn.Conv1d(
            1, config.conv_channels,
            kernel_size=config.conv_kernel_size,
            stride=config.conv_stride,
            padding=config.conv_kernel_size // 2
        )
        
        # Conv blocks for downsampling
        self.conv_blocks = nn.ModuleList()
        in_channels = config.conv_channels
        
        for i, stride in enumerate(config.conv_block_strides):
            out_channels = config.conv_channels * (2 ** min(i, 2))  # Cap at 4x
            self.conv_blocks.append(
                ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride)
            )
            in_channels = out_channels
        
        # Final projection to hidden dimension
        self.projection = nn.Linear(in_channels, config.hidden_dim)
        
        # Positional encoding
        self.pos_conv = nn.Conv1d(
            config.hidden_dim, config.hidden_dim,
            kernel_size=128, padding=64, groups=16
        )
        
        # ALiBi-style positional bias
        self.max_positions = 10000
        slopes = torch.tensor([1.0 / (2 ** (8 * i / config.num_attention_heads)) 
                              for i in range(config.num_attention_heads)])
        self.register_buffer('alibi_slopes', slopes)
        
    def get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Create ALiBi positional bias
        position_ids = torch.arange(seq_len, device=device, dtype=torch.float32)
        position_ids = position_ids.unsqueeze(0) - position_ids.unsqueeze(1)
        
        # Apply slopes
        alibi_bias = position_ids.unsqueeze(0) * self.alibi_slopes.unsqueeze(-1).unsqueeze(-1)
        return alibi_bias
    
    def apply_masking(self, features: torch.Tensor, mask_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask_indices is None and self.training:
            batch_size, seq_len, _ = features.shape
            # Dynamic mask prob schedule
            mask_prob = self.config.mask_prob
            sched = getattr(self.config, 'mask_prob_schedule', None)
            if sched is not None:
                # assume global step stored externally via attribute if set
                global_step = getattr(self, 'global_step', 0)
                # sched: ((step, prob), ... ) sorted
                last_p = mask_prob
                for step, p in sched:
                    if global_step >= step:
                        last_p = p
                mask_prob = last_p
            num_masked = int(seq_len * mask_prob)
            mask_indices = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=features.device)
            min_len, max_len = getattr(self.config, 'mask_length_range', (self.config.mask_length, self.config.mask_length))
            for b in range(batch_size):
                covered = 0
                attempts = 0
                while covered < num_masked and attempts < num_masked * 4:
                    span_len = torch.randint(min_len, max_len+1, (1,)).item()
                    start = torch.randint(0, max(1, seq_len - span_len), (1,)).item()
                    end = min(start + span_len, seq_len)
                    if mask_indices[b, start:end].any():
                        attempts += 1
                        continue
                    mask_indices[b, start:end] = True
                    covered += (end - start)
                    attempts += 1
        
        if mask_indices is not None:
            # Apply masking
            masked_features = features.clone()
            masked_features[mask_indices] = 0
            return masked_features, mask_indices
        
        return features, None
    
    def forward(self, waveform: torch.Tensor, mask_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Input: [batch_size, time_steps] or [batch_size, 1, time_steps]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # Add channel dimension
        
        # Initial convolution
        x = self.initial_conv(waveform)  # [B, conv_channels, T//5]
        
        # Apply conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)  # [B, channels, T//320]
        
        # Transpose for transformer: [B, T, C]
        x = x.transpose(1, 2)
        
        # Project to hidden dimension
        x = self.projection(x)  # [B, T, hidden_dim]
        
        # Add convolutional positional encoding
        pos_x = x.transpose(1, 2)  # [B, hidden_dim, T]
        pos_x = self.pos_conv(pos_x)
        pos_x = pos_x.transpose(1, 2)  # [B, T, hidden_dim]
        # Some kernel/padding combos (even kernel) can produce off-by-one length.
        if pos_x.size(1) > x.size(1):
            pos_x = pos_x[:, :x.size(1)]
        elif pos_x.size(1) < x.size(1):
            # pad at end
            pad_len = x.size(1) - pos_x.size(1)
            pad_tensor = pos_x.new_zeros(pos_x.size(0), pad_len, pos_x.size(2))
            pos_x = torch.cat([pos_x, pad_tensor], dim=1)
        x = x + pos_x
        
        # Apply masking if specified
        x, mask_indices = self.apply_masking(x, mask_indices)
        
        return x, mask_indices
    
    def get_downsample_rate(self) -> int:
        # Calculate total downsampling factor
        total_stride = self.config.conv_stride
        for stride in self.config.conv_block_strides:
            total_stride *= stride
        return total_stride