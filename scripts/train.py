"""
HuMERT-300M Training Script with 2-Stage Curriculum Learning
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import time
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from humert.model import HuMERTModel
from config.model_config import ModelConfig, TrainingConfig, get_model_config, get_training_config
from training.optimizer import create_optimizer_and_scheduler, GradientClipper, MemoryOptimizer, AdapterFreezing, setup_mixed_precision
from training.data_loader import create_data_loaders


class HuMERTTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = get_model_config()
        self.training_config = get_training_config()
        
        # Initialize model
        self.model = HuMERTModel(self.model_config).to(self.device)
        
        # Multi-GPU setup
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        # Freeze teachers to save memory
        self.model.freeze_teachers()
        
        # Setup optimization
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            self.model, self.training_config
        )
        
        self.gradient_clipper = GradientClipper(self.training_config.gradient_clipping)
        
        # Mixed precision setup
        self.scaler, self.autocast = setup_mixed_precision()
        
        # Memory optimization
        MemoryOptimizer.enable_activation_checkpointing(self.model)
        
        # Training state
        self.global_step = 0
        self.current_stage = 1
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup wandb and local logging"""
        if self.config.get('use_wandb', True):
            wandb.init(
                project="humert-300m",
                config={
                    **self.model_config.__dict__,
                    **self.training_config.__dict__,
                    **self.config
                },
                tags=[f"stage_{self.current_stage}"]
            )
            
            # Log model statistics
            wandb.log(self.model.module.get_model_stats() if hasattr(self.model, 'module') else self.model.get_model_stats())
    
    def create_data_loaders_for_stage(self, stage: int) -> tuple:
        """Create data loaders for specific training stage"""
        
        if stage == 1:
            sequence_length = self.training_config.stage1_sequence_length
            batch_size = self.training_config.stage1_batch_size
        elif stage == 2:
            sequence_length = self.training_config.stage2_sequence_length
            batch_size = self.training_config.stage2_batch_size
        else:  # stage 3
            sequence_length = self.training_config.stage3_sequence_length
            batch_size = self.training_config.stage3_batch_size
        
        data_config = {
            'train_paths': self.config['train_paths'],
            'val_paths': self.config['val_paths'],
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'num_workers': self.config.get('num_workers', 4),
            'temperature_alpha': self.training_config.temperature_sampling_alpha,
            'stage': stage
        }
        
        return create_data_loaders(data_config)
    
    def train_step(self, batch: Dict[str, torch.Tensor], stage: int) -> Dict[str, float]:
        """Single training step"""
        
        waveforms = batch['waveforms'].to(self.device)
        language_ids = batch['language_ids'].to(self.device) if 'language_ids' in batch else None
        
        # Determine active heads based on batch content
        active_heads = {
            'dac': True,
            'speech': any(atype == 'speech' for atype in batch.get('audio_types', ['unknown'])),
            'music': any(atype == 'music' for atype in batch.get('audio_types', ['unknown']))
        }
        
        # Forward pass
        if self.scaler is not None:
            with self.autocast():
                outputs = self.model(
                    waveforms, 
                    return_teacher_labels=True,
                    active_heads=active_heads
                )
                loss = outputs['total']
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = self.gradient_clipper.clip_gradients(self.model)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(
                waveforms,
                return_teacher_labels=True, 
                active_heads=active_heads
            )
            loss = outputs['total']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = self.gradient_clipper.clip_gradients(self.model)
            
            # Optimizer step
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        # Collect metrics
        metrics = {
            'total_loss': loss.item(),
            'grad_norm': grad_norm,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        # Add individual task losses
        for key, value in outputs.items():
            if key.endswith('_weighted') and isinstance(value, torch.Tensor):
                metrics[key] = value.item()
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        val_metrics = {}
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                waveforms = batch['waveforms'].to(self.device)
                
                outputs = self.model(waveforms, return_teacher_labels=True)
                batch_size = waveforms.size(0)
                
                # Accumulate losses
                for key, value in outputs.items():
                    if key.endswith('loss') or key == 'total':
                        if key not in val_metrics:
                            val_metrics[key] = 0
                        val_metrics[key] += value.item() * batch_size
                
                total_samples += batch_size
        
        # Average the metrics
        for key in val_metrics:
            val_metrics[key] /= total_samples
        
        self.model.train()
        return val_metrics
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint_data = {
            'global_step': self.global_step,
            'current_stage': self.current_stage,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_checkpoint(filepath, **checkpoint_data)
        
        if is_best:
            best_path = filepath.replace('.pt', '_best.pt')
            model_to_save.save_checkpoint(best_path, **checkpoint_data)
    
    def train_stage(self, stage: int):
        """Train a specific stage"""
        print(f"\n{'='*50}")
        print(f"Starting Stage {stage} Training")
        print(f"{'='*50}")
        
        self.current_stage = stage
        
        # Update wandb tags
        if self.config.get('use_wandb', True):
            wandb.config.update({'current_stage': stage})
        
        # Stage-specific optimizations
        if stage == 3:
            print("Applying Stage 3 optimizations...")
            model_to_optimize = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_optimize.setup_stage3_optimization()
            
            # Recreate optimizer for unfrozen parameters
            self.optimizer, self.scheduler = create_optimizer_and_scheduler(
                self.model, self.training_config
            )
        
        # Create data loaders for this stage
        train_loader, val_loader = self.create_data_loaders_for_stage(stage)
        
        # Determine number of steps for this stage
        if stage == 1:
            target_steps = self.training_config.stage1_steps
        elif stage == 2:
            target_steps = self.training_config.stage2_steps
        else:
            target_steps = self.training_config.stage3_steps
        
        stage_start_step = self.global_step
        
        # Training loop
        pbar = tqdm(total=target_steps, desc=f"Stage {stage}")
        
        while self.global_step - stage_start_step < target_steps:
            for batch in train_loader:
                if self.global_step - stage_start_step >= target_steps:
                    break
                
                # Training step
                metrics = self.train_step(batch, stage)
                
                # Update progress
                self.global_step += 1
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{metrics['total_loss']:.4f}",
                    'lr': f"{metrics['learning_rate']:.2e}",
                    'grad': f"{metrics['grad_norm']:.2f}"
                })
                
                # Logging
                if self.config.get('use_wandb', True):
                    wandb.log({
                        'stage': stage,
                        'step': self.global_step,
                        **metrics,
                        **MemoryOptimizer.get_memory_stats()
                    })
                
                # Validation
                if self.global_step % self.training_config.val_frequency == 0:
                    print("\nRunning validation...")
                    val_metrics = self.validate(val_loader)
                    
                    # Check for best model
                    val_loss = val_metrics.get('total', float('inf'))
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                    
                    # Save checkpoint
                    checkpoint_dir = Path(self.config['output_dir']) / 'checkpoints'
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    checkpoint_path = checkpoint_dir / f'checkpoint_stage_{stage}_step_{self.global_step}.pt'
                    self.save_checkpoint(str(checkpoint_path), is_best)
                    
                    # Log validation metrics
                    if self.config.get('use_wandb', True):
                        wandb.log({
                            'val_' + k: v for k, v in val_metrics.items()
                        })
                    
                    print(f"Validation - Total Loss: {val_loss:.4f} {'(Best!)' if is_best else ''}")
        
        pbar.close()
        print(f"Stage {stage} completed!")
    
    def train(self):
        """Main training function with 2-stage curriculum"""
        print("Starting HuMERT-300M Training")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.module.count_parameters() if hasattr(self.model, 'module') else self.model.count_parameters():,}")
        
        # Stage 1: Warm-up (5 seconds, 60K steps)
        self.train_stage(1)
        
        # Stage 2: Main Training (5 seconds, 240K steps)  
        self.train_stage(2)
        
        # Stage 3: Long-range Spike (8 seconds, 35K steps)
        self.train_stage(3)
        
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Total steps: {self.global_step}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*50)
        
        # Final model save
        final_path = Path(self.config['output_dir']) / 'humert_300m_final.pt'
        self.save_checkpoint(str(final_path))
        
        if self.config.get('use_wandb', True):
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train HuMERT-300M')
    parser.add_argument('--config', type=str, required=True, help='Training configuration file')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['output_dir'] = args.output_dir
    config['use_wandb'] = not args.no_wandb
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = HuMERTTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.global_step = checkpoint['global_step']
        trainer.current_stage = checkpoint['current_stage']
        trainer.epoch = checkpoint['epoch']
        trainer.best_val_loss = checkpoint['best_val_loss']
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()