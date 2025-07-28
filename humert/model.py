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
            teacher_labels = self.generate_teacher_labels(waveform)
            outputs.update(teacher_labels)
            
            # 6. Compute losses
            if self.training:
                losses = self.compute_losses(predictions, teacher_labels, mask_indices)
                outputs.update(losses)
        
        return outputs
    
    def generate_teacher_labels(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate labels from teacher models"""
        teacher_labels = {}
        
        # Generate DAC labels
        with torch.no_grad():
            dac_outputs = self.dac_teacher(waveform)
            teacher_labels['dac_tokens'] = dac_outputs['dac_tokens']
        
        # Generate Speech labels  
        with torch.no_grad():
            speech_outputs = self.speech_teacher(waveform)
            teacher_labels['speech_tokens'] = speech_outputs['speech_tokens']
        
        # Generate Musical labels
        with torch.no_grad():
            music_outputs = self.musical_teacher(waveform)
            teacher_labels['music_features'] = music_outputs['music_features']
        
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
                # All prediction heads
                predictions = self.prediction_heads(encoded_features)
            else:
                # Single task prediction
                predictions = {
                    f'{task}_output': self.prediction_heads.compute_predictions(encoded_features, task)
                }
            
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