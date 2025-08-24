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
        
        # Contrastive projections (shared latent space)
        if getattr(config, 'use_contrastive', False):
            d = config.contrastive_dim
            self.contrastive_proj_audio = nn.Linear(config.hidden_dim, d)
            self.contrastive_proj_speech = nn.Linear(config.hidden_dim, d)
            self.contrastive_proj_music = nn.Linear(config.hidden_dim, d)
            nn.init.xavier_uniform_(self.contrastive_proj_audio.weight)
            nn.init.xavier_uniform_(self.contrastive_proj_speech.weight)
            nn.init.xavier_uniform_(self.contrastive_proj_music.weight)
        
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
        
        # Contrastive projected features (not used for direct supervised loss)
        if getattr(self.config, 'use_contrastive', False):
            ca = F.normalize(self.contrastive_proj_audio(shared_features), dim=-1)
            outputs['contrastive_audio'] = ca
            if self.enable_speech:
                cs = F.normalize(self.contrastive_proj_speech(shared_features), dim=-1)
                outputs['contrastive_speech'] = cs
            if self.enable_music:
                cm = F.normalize(self.contrastive_proj_music(shared_features), dim=-1)
                outputs['contrastive_music'] = cm
        
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

            # --- Base task losses ---
            self.dac_criterion = nn.CrossEntropyLoss(ignore_index=-100)
            self.speech_criterion = nn.CrossEntropyLoss(ignore_index=-100)
            self.music_criterion = nn.MSELoss()

            # --- GradNorm bookkeeping (for 3 supervised tasks) ---
            self.alpha = config.gradnorm_alpha if hasattr(config, 'gradnorm_alpha') else 1.5
            self.register_buffer('task_weights', torch.ones(3))      # dynamic weights for dac / speech / music
            self.register_buffer('initial_losses', torch.zeros(3))   # snapshot of initial losses
            self.register_buffer('loss_ratios', torch.zeros(3))      # not strictly required but kept for logging/debug

            # --- Auxiliary (not reweighted by GradNorm) ---
            self.temperature = getattr(config, 'contrastive_temperature', 0.1)
            self.contrastive_loss_scale = getattr(config, 'contrastive_loss_scale', 1.0)
            self.max_contrastive_samples = getattr(config, 'contrastive_subsample', 4096)
            self.ms_consistency_weight = getattr(config, 'multi_scale_consistency_weight', 0.0)
            # Contrastive enhancements
            self.contrastive_time_pool = getattr(config, 'contrastive_time_pool', True)
            self.contrastive_learn_temp = getattr(config, 'contrastive_learn_temp', False)
            if self.contrastive_learn_temp:
                # reparameterize temperature as exp(log_temp) for positivity
                init_t = float(getattr(config, 'contrastive_temperature', 0.1))
                self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_t)))
            queue_size = getattr(config, 'contrastive_queue_size', 0)
            self.contrastive_queue_size = queue_size
            if queue_size and queue_size > 0:
                dim = getattr(config, 'contrastive_dim', 128)
                self.register_buffer('contrastive_queue_z1', torch.randn(queue_size, dim))
                self.register_buffer('contrastive_queue_z2', torch.randn(queue_size, dim))
                self.register_buffer('contrastive_queue_ptr', torch.zeros(1, dtype=torch.long))
                self.contrastive_queue_z1 = F.normalize(self.contrastive_queue_z1, dim=-1)
                self.contrastive_queue_z2 = F.normalize(self.contrastive_queue_z2, dim=-1)
        
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
        task_order = ['dac', 'speech', 'music']
        valid_tasks = [t for t in task_order if t in losses]
        if len(valid_tasks) < 2:
            return
        # Compute per-task grad norm
        grad_norms = []
        current_losses = []
        for t in task_order:
            if t in losses:
                grad = torch.autograd.grad(losses[t], shared_params, retain_graph=True, create_graph=True)[0]
                grad_norms.append(grad.norm())
                current_losses.append(losses[t].detach())
            else:
                grad_norms.append(torch.tensor(0.0, device=shared_params.device))
                current_losses.append(torch.tensor(0.0, device=shared_params.device))
        grad_norms = torch.stack(grad_norms)
        current_losses = torch.stack(current_losses)
        if self.initial_losses.sum() == 0:
            self.initial_losses = current_losses.detach().clamp_min(1e-8)
        loss_ratios = (current_losses / self.initial_losses).clamp_min(1e-6)
        mean_loss_ratio = loss_ratios[loss_ratios>0].mean()
        mean_grad_norm = grad_norms[grad_norms>0].mean().detach()
        targets = mean_grad_norm * (loss_ratios / mean_loss_ratio) ** self.alpha
        new_weights = []
        for i, t in enumerate(task_order):
            if t in losses and grad_norms[i] > 0:
                w = self.task_weights[i] * (targets[i] / (grad_norms[i] + 1e-8))
            else:
                w = self.task_weights[i]
            new_weights.append(w)
        new_weights = torch.stack(new_weights)
        # Normalize weights (avoid collapse)
        denom = new_weights.sum().clamp_min(1e-6)
        self.task_weights = (new_weights / denom) * len(task_order)
    
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
        
        # Contrastive loss (optional, separate from GradNorm reweighting)
        if getattr(self.config, 'use_contrastive', False) and 'contrastive_audio' in predictions:
            ca = predictions['contrastive_audio']  # [B,T,D]
            # Prefer speech as second view else music
            cb = None
            if 'contrastive_speech' in predictions:
                cb = predictions['contrastive_speech']
            elif 'contrastive_music' in predictions:
                cb = predictions['contrastive_music']
            if cb is not None:
                # Temporal pooling (default) to stabilize and reduce compute
                if self.contrastive_time_pool:
                    # mean over time; if mask provided and boolean, use masked subset
                    if mask is not None and mask.any():
                        # mask shape [B,T], expand for weighting
                        mfloat = mask.float()
                        z1p = (ca * mfloat.unsqueeze(-1)).sum(1) / (mfloat.sum(1, keepdim=True) + 1e-6)
                        z2p = (cb * mfloat.unsqueeze(-1)).sum(1) / (mfloat.sum(1, keepdim=True) + 1e-6)
                    else:
                        z1p = ca.mean(1)
                        z2p = cb.mean(1)
                else:
                    # sample time steps as before
                    B, T, D = ca.shape
                    if mask is not None and mask.any():
                        idx = mask.nonzero(as_tuple=False)  # [N,2]
                    else:
                        b_idx = torch.randint(0, B, (min(self.max_contrastive_samples, B*T),), device=ca.device)
                        t_idx = torch.randint(0, T, (b_idx.size(0),), device=ca.device)
                        idx = torch.stack([b_idx, t_idx], dim=1)
                    if idx.size(0) > self.max_contrastive_samples:
                        choice = torch.randperm(idx.size(0), device=ca.device)[:self.max_contrastive_samples]
                        idx = idx[choice]
                    z1p = ca[idx[:,0], idx[:,1]]
                    z2p = cb[idx[:,0], idx[:,1]]

                z1p = F.normalize(z1p, dim=-1)
                z2p = F.normalize(z2p, dim=-1)

                # Temperature (learnable optional)
                temp = torch.exp(self.log_temperature) if hasattr(self, 'log_temperature') else torch.tensor(self.temperature, device=z1p.device)

                # Base similarities (batch positives first)
                logits_ab_targets = z1p @ z2p.t()
                logits_ba_targets = z2p @ z1p.t()

                # Append queue negatives if available
                if self.contrastive_queue_size:
                    qz2 = self.contrastive_queue_z2.detach()
                    qz1 = self.contrastive_queue_z1.detach()
                    logits_ab_neg = z1p @ qz2.t()  # [B,Q]
                    logits_ba_neg = z2p @ qz1.t()  # [B,Q]
                    logits_ab = torch.cat([logits_ab_targets, logits_ab_neg], dim=1) / temp
                    logits_ba = torch.cat([logits_ba_targets, logits_ba_neg], dim=1) / temp
                    labels = torch.arange(z1p.size(0), device=z1p.device, dtype=torch.long)
                else:
                    logits_ab = logits_ab_targets / temp
                    logits_ba = logits_ba_targets / temp
                    labels = torch.arange(z1p.size(0), device=z1p.device, dtype=torch.long)

                loss_ab = F.cross_entropy(logits_ab, labels)
                loss_ba = F.cross_entropy(logits_ba, labels)
                c_loss = 0.5 * (loss_ab + loss_ba)
                losses['contrastive'] = c_loss

                # Update queues (FIFO) if enabled
                if self.contrastive_queue_size:
                    with torch.no_grad():
                        bsz = z1p.size(0)
                        ptr = int(self.contrastive_queue_ptr.item())
                        k = self.contrastive_queue_size
                        # If batch larger than queue, keep last k entries
                        if bsz >= k:
                            self.contrastive_queue_z1.copy_(z1p[-k:])
                            self.contrastive_queue_z2.copy_(z2p[-k:])
                            self.contrastive_queue_ptr.zero_()
                        else:
                            end = ptr + bsz
                            if end <= k:
                                self.contrastive_queue_z1[ptr:end] = z1p
                                self.contrastive_queue_z2[ptr:end] = z2p
                            else:
                                first = k - ptr
                                self.contrastive_queue_z1[ptr:] = z1p[:first]
                                self.contrastive_queue_z1[:bsz-first] = z1p[first:]
                                self.contrastive_queue_z2[ptr:] = z2p[:first]
                                self.contrastive_queue_z2[:bsz-first] = z2p[first:]
                            self.contrastive_queue_ptr[0] = (end % k)
        
        # Update GradNorm weights if shared parameters provided
        if shared_params is not None and len(losses) > 1:
            self.update_gradnorm_weights(losses, shared_params)
        
        # Compute weighted total loss
        total_loss = 0.0
        task_names = ['dac', 'speech', 'music']
        for i, task in enumerate(task_names):
            if task in losses:
                weighted = self.task_weights[i] * losses[task]
                losses[f'{task}_weighted'] = weighted
                losses[f'{task}_weight'] = self.task_weights[i].detach()
                total_loss += weighted
        # Add contrastive (outside GradNorm weighting)
        if 'contrastive' in losses:
            scaled = losses['contrastive'] * self.contrastive_loss_scale
            losses['contrastive_weighted'] = scaled
            total_loss += scaled
        # Multi-scale consistency (outside GradNorm)
        if self.ms_consistency_weight > 0 and 'cqt_features_scale1' in targets and 'cqt_features_scale2' in targets:
            f1 = targets['cqt_features_scale1']
            f2 = targets['cqt_features_scale2']
            # Align time length
            if f1.size(1) != f2.size(1):
                if f1.size(1) < f2.size(1):
                    f1 = F.interpolate(f1.transpose(1,2), size=f2.size(1), mode='linear', align_corners=False).transpose(1,2)
                else:
                    f2 = F.interpolate(f2.transpose(1,2), size=f1.size(1), mode='linear', align_corners=False).transpose(1,2)
            diff = (F.normalize(f1, dim=-1) - F.normalize(f2, dim=-1))**2
            ms_loss = diff.mean()
            losses['music_ms_consistency'] = ms_loss
            losses['music_ms_consistency_weighted'] = ms_loss * self.ms_consistency_weight
            total_loss += losses['music_ms_consistency_weighted']
        losses['total'] = total_loss if isinstance(total_loss, torch.Tensor) else torch.tensor(total_loss, device=shared_params.device if shared_params is not None else next(iter(predictions.values())).device)
        losses['task_weights'] = self.task_weights.detach().clone()
        return losses