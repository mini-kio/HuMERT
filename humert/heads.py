"""
Prediction Heads with Adapter Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from config.model_config import ModelConfig


class Adapter(nn.Module):
    def __init__(self, hidden_dim: int, adapter_dim: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x


class DACPredictionHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_codebooks = config.dac_codebooks
        self.vocab_size = config.dac_vocab_size
        
        # Adapter for task-specific adaptation
        self.adapter = Adapter(config.hidden_dim, config.adapter_dim)
        
        # Prediction head for 9 codebooks
        self.prediction_head = nn.Linear(config.hidden_dim, self.num_codebooks * self.vocab_size)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.prediction_head.weight)
        nn.init.zeros_(self.prediction_head.bias)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply adapter
        adapted_states = self.adapter(hidden_states)  # [B, T, hidden_dim]
        
        # Predict for all codebooks
        logits = self.prediction_head(adapted_states)  # [B, T, 9*1024]
        
        # Reshape to separate codebooks
        batch_size, seq_len, _ = logits.shape
        logits = logits.view(batch_size, seq_len, self.num_codebooks, self.vocab_size)
        
        return logits  # [B, T, 9, 1024]


class SpeechPredictionHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_clusters = config.speech_clusters
        
        # Adapter for task-specific adaptation
        self.adapter = Adapter(config.hidden_dim, config.adapter_dim)
        
        # Prediction head for speech clusters
        self.prediction_head = nn.Linear(config.hidden_dim, self.num_clusters)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.prediction_head.weight)
        nn.init.zeros_(self.prediction_head.bias)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply adapter
        adapted_states = self.adapter(hidden_states)  # [B, T, hidden_dim]
        
        # Predict speech clusters
        logits = self.prediction_head(adapted_states)  # [B, T, 1500]
        
        return logits


class MusicPredictionHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.output_dim = config.cqt_bins
        
        # Adapter for task-specific adaptation
        self.adapter = Adapter(config.hidden_dim, config.adapter_dim)
        
        # Prediction head for musical features (continuous)
        self.prediction_head = nn.Linear(config.hidden_dim, self.output_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.prediction_head.weight)
        nn.init.zeros_(self.prediction_head.bias)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Apply adapter
        adapted_states = self.adapter(hidden_states)  # [B, T, hidden_dim]
        
        # Predict musical features
        features = self.prediction_head(adapted_states)  # [B, T, 84]
        
        return features


class PredictionHeads(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Shared projection (common to all tasks)
        self.shared_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Task-specific prediction heads
        self.dac_head = DACPredictionHead(config)
        self.speech_head = SpeechPredictionHead(config)
        self.music_head = MusicPredictionHead(config)
        
        # Gradient masking flags
        self.enable_dac = True
        self.enable_speech = True
        self.enable_music = True
        
    def set_active_heads(self, dac: bool = True, speech: bool = True, music: bool = True):
        """Control which heads are active for gradient computation"""
        self.enable_dac = dac
        self.enable_speech = speech
        self.enable_music = music
    
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Apply shared projection
        shared_features = self.shared_projection(hidden_states)  # [B, T, hidden_dim]
        
        outputs = {}
        
        # DAC prediction head
        if self.enable_dac:
            dac_logits = self.dac_head(shared_features)
            outputs['dac_logits'] = dac_logits
        
        # Speech prediction head
        if self.enable_speech:
            speech_logits = self.speech_head(shared_features)
            outputs['speech_logits'] = speech_logits
        
        # Music prediction head
        if self.enable_music:
            music_features = self.music_head(shared_features)
            outputs['music_features'] = music_features
        
        return outputs
    
    def compute_predictions(self, hidden_states: torch.Tensor, task: str) -> torch.Tensor:
        """Compute predictions for a specific task only"""
        shared_features = self.shared_projection(hidden_states)
        
        if task == 'dac':
            return self.dac_head(shared_features)
        elif task == 'speech':
            return self.speech_head(shared_features)
        elif task == 'music':
            return self.music_head(shared_features)
        else:
            raise ValueError(f"Unknown task: {task}")


class MultiTaskLoss(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Loss functions
        self.dac_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.speech_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.music_criterion = nn.MSELoss()
        
        # GradNorm parameters
        self.alpha = config.gradnorm_alpha if hasattr(config, 'gradnorm_alpha') else 1.5
        self.register_buffer('task_weights', torch.ones(3))  # 3 tasks
        self.register_buffer('initial_losses', torch.zeros(3))
        self.register_buffer('loss_ratios', torch.zeros(3))
        
    def compute_dac_loss(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute DAC reconstruction loss for all codebooks"""
        batch_size, seq_len, num_codebooks, vocab_size = logits.shape
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, vocab_size)  # [B*T*9, vocab_size]
        targets_flat = targets.view(-1)  # [B*T*9]
        
        if mask is not None:
            # Apply mask
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_codebooks).contiguous()
            mask_flat = mask_expanded.view(-1)
            logits_flat = logits_flat[mask_flat]
            targets_flat = targets_flat[mask_flat]
        
        return self.dac_criterion(logits_flat, targets_flat)
    
    def compute_speech_loss(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute speech clustering loss"""
        logits_flat = logits.view(-1, logits.size(-1))  # [B*T, num_clusters]
        targets_flat = targets.view(-1)  # [B*T]
        
        if mask is not None:
            mask_flat = mask.view(-1)
            logits_flat = logits_flat[mask_flat]
            targets_flat = targets_flat[mask_flat]
        
        return self.speech_criterion(logits_flat, targets_flat)
    
    def compute_music_loss(self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute music feature regression loss"""
        if mask is not None:
            # Apply mask to both predictions and targets
            predictions = predictions[mask]
            targets = targets[mask]
        else:
            # Flatten for MSE loss
            predictions = predictions.view(-1, predictions.size(-1))
            targets = targets.view(-1, targets.size(-1))
        
        return self.music_criterion(predictions, targets)
    
    def update_gradnorm_weights(self, losses: Dict[str, torch.Tensor], shared_params: torch.Tensor):
        """Update task weights using GradNorm algorithm"""
        # Compute gradients w.r.t shared parameters
        task_grads = {}
        for task, loss in losses.items():
            if loss.requires_grad:
                grad = torch.autograd.grad(loss, shared_params, retain_graph=True, create_graph=True)[0]
                task_grads[task] = torch.norm(grad)
        
        if len(task_grads) > 1:
            # Compute relative loss ratios
            current_losses = torch.tensor([losses[task].item() for task in ['dac', 'speech', 'music']])
            
            if self.initial_losses.sum() == 0:
                self.initial_losses = current_losses.clone()
            
            loss_ratios = current_losses / (self.initial_losses + 1e-8)
            
            # Compute gradient norms
            grad_norms = torch.tensor([task_grads.get(task, 0.0) for task in ['dac', 'speech', 'music']])
            
            # Update weights using GradNorm
            mean_grad_norm = grad_norms.mean()
            mean_loss_ratio = loss_ratios.mean()
            
            targets = mean_grad_norm * (loss_ratios / mean_loss_ratio) ** self.alpha
            
            # Update task weights
            for i, (task, target) in enumerate(zip(['dac', 'speech', 'music'], targets)):
                if task in task_grads:
                    self.task_weights[i] = self.task_weights[i] * (target / (grad_norms[i] + 1e-8))
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], 
                mask: Optional[torch.Tensor] = None, shared_params: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # Compute individual losses
        if 'dac_logits' in predictions and 'dac_tokens' in targets:
            losses['dac'] = self.compute_dac_loss(predictions['dac_logits'], targets['dac_tokens'], mask)
        
        if 'speech_logits' in predictions and 'speech_tokens' in targets:
            losses['speech'] = self.compute_speech_loss(predictions['speech_logits'], targets['speech_tokens'], mask)
        
        if 'music_features' in predictions and 'music_features' in targets:
            losses['music'] = self.compute_music_loss(predictions['music_features'], targets['music_features'], mask)
        
        # Update GradNorm weights if shared parameters provided
        if shared_params is not None and len(losses) > 1:
            self.update_gradnorm_weights(losses, shared_params)
        
        # Compute weighted total loss
        total_loss = 0
        task_names = ['dac', 'speech', 'music']
        for i, task in enumerate(task_names):
            if task in losses:
                weighted_loss = self.task_weights[i] * losses[task]
                losses[f'{task}_weighted'] = weighted_loss
                total_loss += weighted_loss
        
        losses['total'] = total_loss
        
        return losses