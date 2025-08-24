"""
Training optimization utilities
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Dict, Any, Optional


class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)
    
    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            # Linear warmup
            warmup_factor = self._step_count / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self._step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            lr_factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor
            return [base_lr * lr_factor for base_lr in self.base_lrs]


def create_optimizer_and_scheduler(model: nn.Module, config) -> tuple:
    """Create optimizer and learning rate scheduler"""
    
    # Separate parameters for different components
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    
    # Create AdamW optimizer with bitsandbytes for fp8 optimizer states
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=config.betas,
            eps=1e-8
        )
    except ImportError:
        print("Warning: bitsandbytes not available, using standard AdamW")
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=config.betas,
            eps=1e-8
        )
    
    # Create cosine warmup scheduler
    total_steps = getattr(config, 'total_steps', None)
    if total_steps is None or total_steps == 0:
        # Fallback: infer from known stage attributes if present
        stage_attrs = [getattr(config, n, 0) for n in ['stage1_steps','stage2_steps','stage3_steps']]
        total_steps = sum(stage_attrs) if any(stage_attrs) else 100000
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=getattr(config, 'warmup_steps', 8000),
        total_steps=total_steps,
        min_lr_ratio=getattr(config, 'min_lr_ratio', 0.1)
    )
    
    return optimizer, scheduler


class GradientClipper:
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients and return the gradient norm"""
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        if len(parameters) == 0:
            return 0.0
        
        # Compute gradient norm
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in parameters])
        ).item()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)
        
        return total_norm


class MemoryOptimizer:
    """Memory optimization utilities for training"""
    
    @staticmethod
    def enable_activation_checkpointing(model: nn.Module):
        """Enable gradient checkpointing to save memory"""
        if hasattr(model, 'encoder'):
            for layer in model.encoder.layers:
                if hasattr(layer, 'gradient_checkpointing'):
                    layer.gradient_checkpointing = True
    
    @staticmethod
    def setup_zero_optimization(model: nn.Module, optimizer, stage: int = 2):
        """Setup ZeRO optimization if DeepSpeed is available"""
        try:
            import deepspeed  # noqa: F401
            
            # ZeRO-2 configuration
            zero_config = {
                "stage": stage,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            }
            
            return deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config_params={
                    "zero_optimization": zero_config,
                    "fp16": {"enabled": True},
                    "gradient_clipping": 1.0
                }
            )
        except ImportError:
            print("Warning: DeepSpeed not available, skipping ZeRO optimization")
            return model, optimizer
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current GPU memory statistics"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_allocated_gb": max_allocated
            }
        return {}


class AdapterFreezing:
    """Utilities for freezing/unfreezing model components during training"""
    
    @staticmethod
    def freeze_early_layers(model: nn.Module, freeze_layers: int = 16):
        """Freeze early transformer layers (Stage 3 optimization)"""
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            for i, layer in enumerate(model.encoder.layers[:freeze_layers]):
                for param in layer.parameters():
                    param.requires_grad = False
                print(f"Frozen layer {i}")
    
    @staticmethod
    def unfreeze_all_layers(model: nn.Module):
        """Unfreeze all model parameters"""
        for param in model.parameters():
            param.requires_grad = True
    
    @staticmethod
    def get_trainable_parameters(model: nn.Module) -> int:
        """Count number of trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_mixed_precision():
    """Setup mixed precision training"""
    try:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        return scaler, autocast
    except ImportError:
        print("Warning: Mixed precision not available")
        return None, None