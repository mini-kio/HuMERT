"""
Multi-Teacher System: DAC, Speech, and Musical Teachers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Optional, Tuple, Dict, Any
from config.model_config import ModelConfig


class DACTeacher(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_codebooks = config.dac_codebooks
        self.vocab_size = config.dac_vocab_size
        self.frame_rate = config.dac_frame_rate
        
        # Simulated DAC encoder (in real implementation, use pretrained DAC)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 3, stride=2, padding=1),
        )
        
        # Quantization layers for each codebook
        self.quantizers = nn.ModuleList([
            nn.Linear(1024, self.vocab_size) for _ in range(self.num_codebooks)
        ])
        
    def encode_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        # Input: [B, T_audio] -> [B, 1, T_audio]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        
        # Encode to latent representation
        latents = self.encoder(waveform)  # [B, 1024, T_latent]
        return latents.transpose(1, 2)  # [B, T_latent, 1024]
    
    def quantize(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = latents.shape
        
        # Quantize with each codebook
        quantized_indices = []
        quantized_features = []
        
        for i, quantizer in enumerate(self.quantizers):
            # Project to vocabulary space
            logits = quantizer(latents)  # [B, T, vocab_size]
            
            # Gumbel softmax for differentiable quantization during training
            if self.training:
                indices = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
                indices_hard = indices.argmax(dim=-1)
            else:
                indices_hard = logits.argmax(dim=-1)
                indices = F.one_hot(indices_hard, self.vocab_size).float()
            
            quantized_indices.append(indices_hard)
            quantized_features.append(indices)
        
        # Stack codebooks: [B, T, num_codebooks]
        indices = torch.stack(quantized_indices, dim=2)
        features = torch.stack(quantized_features, dim=2)  # [B, T, num_codebooks, vocab_size]
        
        return indices, features
    
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        latents = self.encode_audio(waveform)
        indices, features = self.quantize(latents)
        
        return {
            'dac_tokens': indices,  # [B, T, 9]
            'dac_features': features,  # [B, T, 9, 1024]
            'dac_latents': latents
        }


class SpeechTeacher(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_clusters = config.speech_clusters
        self.num_languages = config.speech_languages
        
        # Feature extractors
        self.mel_extractor = nn.Conv1d(1, 80, 400, stride=160, padding=200)  # 80 mel bins
        self.mfcc_extractor = nn.Conv1d(80, 13, 1)  # 13 MFCC coefficients
        
        # Clustering projection
        self.feature_proj = nn.Linear(93, 256)  # 80 mel + 13 MFCC
        self.cluster_proj = nn.Linear(256, self.num_clusters)
        
        # Language embedding for multilingual support
        self.language_embedding = nn.Embedding(self.num_languages, 64)
        
        # Two-level upsampling with temperature
        self.temperature_alpha = 0.7
        
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        # Input: [B, T_audio] -> [B, 1, T_audio]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        
        # Extract log mel features
        mel_features = self.mel_extractor(waveform)  # [B, 80, T]
        mel_features = torch.log(mel_features.clamp(min=1e-8))
        
        # Extract MFCC features
        mfcc_features = self.mfcc_extractor(mel_features)  # [B, 13, T]
        
        # Combine features
        combined = torch.cat([mel_features, mfcc_features], dim=1)  # [B, 93, T]
        return combined.transpose(1, 2)  # [B, T, 93]
    
    def cluster_features(self, features: torch.Tensor, language_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project features
        projected = self.feature_proj(features)  # [B, T, 256]
        
        # Add language embedding if provided
        if language_ids is not None:
            lang_emb = self.language_embedding(language_ids)  # [B, 64]
            lang_emb = lang_emb.unsqueeze(1).expand(-1, projected.size(1), -1)
            projected = torch.cat([projected, lang_emb], dim=-1)  # [B, T, 320]
            projected = nn.Linear(320, 256).to(projected.device)(projected)
        
        # Cluster assignment
        cluster_logits = self.cluster_proj(projected)  # [B, T, num_clusters]
        
        # Apply temperature scaling for two-level upsampling
        cluster_probs = F.softmax(cluster_logits / self.temperature_alpha, dim=-1)
        
        return cluster_logits, cluster_probs
    
    def forward(self, waveform: torch.Tensor, language_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        features = self.extract_features(waveform)
        cluster_logits, cluster_probs = self.cluster_features(features, language_ids)
        
        # Hard assignment for discrete tokens
        cluster_tokens = cluster_logits.argmax(dim=-1)  # [B, T]
        
        return {
            'speech_tokens': cluster_tokens,
            'speech_probs': cluster_probs,
            'speech_features': features
        }


class MusicalTeacher(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.cqt_bins = config.cqt_bins
        self.chroma_bins = config.chroma_bins
        
        # CQT parameters
        self.hop_length = 512
        self.sample_rate = 24000
        
        # Feature processing networks
        self.cqt_processor = nn.Sequential(
            nn.Linear(self.cqt_bins, 128),
            nn.ReLU(),
            nn.Linear(128, self.cqt_bins)
        )
        
        self.chroma_processor = nn.Sequential(
            nn.Linear(self.chroma_bins, 32),
            nn.ReLU(),
            nn.Linear(32, self.chroma_bins)
        )
        
        # Combined output projection
        self.output_proj = nn.Linear(self.cqt_bins + self.chroma_bins, self.cqt_bins)
        
    def extract_cqt_features(self, waveform: torch.Tensor) -> torch.Tensor:
        batch_size = waveform.shape[0]
        cqt_features = []
        
        for i in range(batch_size):
            # Convert to numpy for librosa
            audio_np = waveform[i].cpu().numpy()
            
            # Compute Constant-Q Transform
            cqt = librosa.cqt(
                audio_np,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_bins=self.cqt_bins,
                bins_per_octave=12
            )
            
            # Convert to magnitude and log scale
            cqt_mag = np.abs(cqt)
            cqt_log = np.log(cqt_mag + 1e-8)
            
            cqt_features.append(torch.tensor(cqt_log.T, device=waveform.device))  # [T, bins]
        
        return torch.stack(cqt_features, dim=0)  # [B, T, bins]
    
    def extract_chroma_features(self, waveform: torch.Tensor) -> torch.Tensor:
        batch_size = waveform.shape[0]
        chroma_features = []
        
        for i in range(batch_size):
            # Convert to numpy for librosa
            audio_np = waveform[i].cpu().numpy()
            
            # Compute chroma features
            chroma = librosa.feature.chroma_cqt(
                y=audio_np,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                n_chroma=self.chroma_bins
            )
            
            chroma_features.append(torch.tensor(chroma.T, device=waveform.device))  # [T, 12]
        
        return torch.stack(chroma_features, dim=0)  # [B, T, 12]
    
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract musical features
        cqt_features = self.extract_cqt_features(waveform)  # [B, T, 84]
        chroma_features = self.extract_chroma_features(waveform)  # [B, T, 12]
        
        # Process features through networks
        processed_cqt = self.cqt_processor(cqt_features)
        processed_chroma = self.chroma_processor(chroma_features)
        
        # Combine features
        combined = torch.cat([processed_cqt, processed_chroma], dim=-1)  # [B, T, 96]
        output_features = self.output_proj(combined)  # [B, T, 84]
        
        return {
            'music_features': output_features,
            'cqt_features': cqt_features,
            'chroma_features': chroma_features
        }