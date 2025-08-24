"""
HuMERT-300M: Main model class integrating all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any

from .encoder import FlashLinearTransformerEncoder
from .frontend import ConvFrontend24k
from .teachers import DACTeacher, SpeechTeacher, MusicalTeacher
from .heads import PredictionHeads, MultiTaskLoss
from config.model_config import ModelConfig


class HuMERTModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.frontend = ConvFrontend24k(config)
        self.encoder = FlashLinearTransformerEncoder(config)
        self.prediction_heads = PredictionHeads(config)
        
        # Teacher models (for label generation)
        self.dac_teacher = DACTeacher(config)
        self.speech_teacher = SpeechTeacher(config)
        self.musical_teacher = MusicalTeacher(config)
        
        # Loss function
        self.criterion = MultiTaskLoss(config)
        
        # Model statistics  
        self.total_params = self.count_parameters()
        print(f"HuMERT-300M initialized with {self.total_params/1e6:.1f}M parameters")
        
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, 
                waveform: torch.Tensor, 
                mask_indices: Optional[torch.Tensor] = None,
                return_teacher_labels: bool = True,
                active_heads: Optional[Dict[str, bool]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of HuMERT model
        
        Args:
            waveform: Raw audio waveform [B, T_audio]
            mask_indices: Optional masking indices [B, T_seq]
            return_teacher_labels: Whether to compute teacher labels
            active_heads: Which prediction heads to activate
        
        Returns:
            Dictionary with predictions and losses
        """
        
        # 1. Frontend processing
        features, mask_indices = self.frontend(waveform, mask_indices)  # [B, T_seq, hidden_dim]
        
        # 2. Encoder processing
        encoded_features = self.encoder(features)  # [B, T_seq, hidden_dim]
        
        # 3. Set active prediction heads
        if active_heads is not None:
            self.prediction_heads.set_active_heads(**active_heads)
        
        # 4. Generate predictions
        predictions = self.prediction_heads(encoded_features)
        
        outputs = {
            'encoded_features': encoded_features,
            'mask_indices': mask_indices,
            **predictions
        }
        
        # 5. Generate teacher labels if requested
        if return_teacher_labels:
            teacher_labels = self.generate_teacher_labels(waveform, encoded_features.size(1))
            outputs.update(teacher_labels)
            
            # 6. Compute losses
            if self.training:
                losses = self.compute_losses(predictions, teacher_labels, mask_indices)
                outputs.update(losses)
        
        return outputs
    
    def generate_teacher_labels(self, waveform: torch.Tensor, target_len: int) -> Dict[str, torch.Tensor]:
        """Generate and length-align teacher labels to encoder sequence length.

        Args:
            waveform: [B, T_audio]
            target_len: encoder time steps
        Returns:
            Dict with aligned teacher targets (dac_tokens [B,target_len,num_codebooks], etc.)
        """
        teacher_labels: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            dac_outputs = self.dac_teacher(waveform)
            speech_outputs = self.speech_teacher(waveform)
            music_outputs = self.musical_teacher(waveform)

            audio_len = waveform.shape[-1]
            frontend_hop = self.frontend.get_downsample_rate()

            # DAC alignment
            dac_tokens_raw = dac_outputs['dac_tokens']  # [B, Td, C]
            B0, Td, Ccode = dac_tokens_raw.shape
            dac_hop = audio_len / Td if Td > 0 else frontend_hop
            t_audio_positions = torch.arange(target_len, device=waveform.device).float() * frontend_hop + frontend_hop / 2
            if Td > 0:
                dac_frame_indices = (t_audio_positions / dac_hop).round().clamp_(0, Td - 1).long()
                dac_tokens = dac_tokens_raw.index_select(1, dac_frame_indices)
            else:
                dac_tokens = dac_tokens_raw.new_zeros(B0, target_len, Ccode)

            # Speech alignment
            speech_raw = speech_outputs['speech_tokens']  # [B, Ts]
            Ts = speech_raw.size(1)
            if Ts > 0:
                speech_hop = audio_len / Ts
                speech_indices = (t_audio_positions / speech_hop).round().clamp_(0, Ts - 1).long()
                speech_tokens = speech_raw.index_select(1, speech_indices)
            else:
                speech_tokens = speech_raw.new_zeros(B0, target_len)

            # Music alignment (continuous)
            music_features = music_outputs['music_features']  # [B, Tm, F]
            Bm, Tm, Fm = music_features.shape
            if Tm > 0:
                music_hop = audio_len / Tm
                music_indices = (t_audio_positions / music_hop).round().clamp_(0, Tm - 1).long()
                music_features_interp = music_features.index_select(1, music_indices)
            else:
                music_features_interp = music_features.new_zeros(Bm, target_len, Fm)

            # Optional smoothing (simple moving average) on continuous features
            k = getattr(self.config, 'alignment_smooth_kernel', 0)
            if k and k > 1 and k % 2 == 1:
                pad = k // 2
                # depthwise conv over time per channel
                Bc, Lc, Fc = music_features_interp.shape
                w = torch.ones(Fc, 1, k, device=music_features_interp.device) / k
                mv = F.conv1d(
                    music_features_interp.transpose(1, 2), w, padding=pad, groups=Fc
                ).transpose(1, 2)
                music_features_interp = mv

            teacher_labels['dac_tokens'] = dac_tokens
            teacher_labels['speech_tokens'] = speech_tokens
            teacher_labels['music_features'] = music_features_interp

            if getattr(self.config, 'alignment_return_indices', False):
                teacher_labels['dac_frame_indices'] = dac_frame_indices
                teacher_labels['speech_frame_indices'] = speech_indices if 'speech_indices' in locals() else torch.empty(0, dtype=torch.long)
                teacher_labels['music_frame_indices'] = music_indices if 'music_indices' in locals() else torch.empty(0, dtype=torch.long)

        return teacher_labels
    
    def compute_losses(self, 
                      predictions: Dict[str, torch.Tensor],
                      targets: Dict[str, torch.Tensor], 
                      mask_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute multi-task losses"""
        
        # Get shared parameters for GradNorm
        shared_params = self.prediction_heads.shared_projection.weight
        
        # Compute losses
        losses = self.criterion(predictions, targets, mask_indices, shared_params)
        
        return losses
    
    def inference(self, waveform: torch.Tensor, task: str = 'all') -> Dict[str, torch.Tensor]:
        """Inference mode without teacher label generation"""
        self.eval()
        
        with torch.no_grad():
            # Frontend and encoder
            features, _ = self.frontend(waveform, mask_indices=None)
            encoded_features = self.encoder(features)
            
            if task == 'all':
                predictions = self.prediction_heads(encoded_features)
            else:
                single = self.prediction_heads.compute_predictions(encoded_features, task)
                if task == 'dac':
                    predictions = {'dac_logits': single}
                elif task == 'speech':
                    predictions = {'speech_logits': single}
                elif task == 'music':
                    predictions = {'music_features': single}
                else:
                    raise ValueError("Unknown task")
            predictions['encoded_features'] = encoded_features
            
        return predictions
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract encoded features without prediction heads"""
        self.eval()
        
        with torch.no_grad():
            features, _ = self.frontend(waveform, mask_indices=None)
            encoded_features = self.encoder(features)
            
        return encoded_features
    
    def freeze_teachers(self):
        """Freeze teacher models to save memory during training"""
        for param in self.dac_teacher.parameters():
            param.requires_grad = False
        for param in self.speech_teacher.parameters():
            param.requires_grad = False  
        for param in self.musical_teacher.parameters():
            param.requires_grad = False
    
    def setup_stage3_optimization(self):
        """Setup optimizations for Stage 3 (long-range training)"""
        # Freeze early encoder layers (0-15)
        for i, layer in enumerate(self.encoder.layers[:16]):
            for param in layer.parameters():
                param.requires_grad = False
            print(f"Frozen encoder layer {i}")
        
        # Keep adapter heads trainable
        for param in self.prediction_heads.parameters():
            param.requires_grad = True
        
        print(f"Stage 3: {self.count_trainable_parameters()/1e6:.1f}M trainable parameters")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            'total_parameters': self.total_params,
            'trainable_parameters': self.count_trainable_parameters(),
            'model_size_mb': self.total_params * 4 / 1024 / 1024,  # Assuming fp32
            'encoder_layers': self.config.num_layers,
            'hidden_dim': self.config.hidden_dim,
            'attention_heads': self.config.num_attention_heads,
            'dac_codebooks': self.config.dac_codebooks,
            'speech_clusters': self.config.speech_clusters,
            'cqt_bins': self.config.cqt_bins
        }
    
    def save_checkpoint(self, filepath: str, optimizer_state: Optional[Dict] = None, 
                       scheduler_state: Optional[Dict] = None, step: int = 0):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'step': step,
            'model_stats': self.get_model_stats()
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
            
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = 'cpu') -> Tuple['HuMERTModel', Dict]:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model with config
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint