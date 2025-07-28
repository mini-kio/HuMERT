"""
Data loading and preprocessing for HuMERT training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any
import webdataset as wds
from pathlib import Path


class AudioDataset(Dataset):
    def __init__(self, 
                 data_paths: List[str],
                 sample_rate: int = 24000,
                 sequence_length: float = 5.0,
                 temperature_alpha: float = 0.7,
                 stage: int = 1):
        
        self.data_paths = data_paths
        self.sample_rate = sample_rate
        self.sequence_length = sequence_length
        self.sequence_samples = int(sequence_length * sample_rate)
        self.temperature_alpha = temperature_alpha
        self.stage = stage
        
        # Load metadata
        self.audio_files = []
        self.load_metadata()
        
        # Temperature sampling weights
        self.setup_temperature_sampling()
    
    def load_metadata(self):
        """Load audio file paths and metadata"""
        for data_path in self.data_paths:
            path = Path(data_path)
            if path.is_dir():
                # Recursively find audio files
                audio_extensions = ['.wav', '.flac', '.mp3', '.m4a']
                for ext in audio_extensions:
                    self.audio_files.extend(list(path.rglob(f'*{ext}')))
        
        print(f"Loaded {len(self.audio_files)} audio files")
    
    def setup_temperature_sampling(self):
        """Setup temperature sampling for balanced training"""
        # Categorize files by type (simplified heuristic)
        self.speech_files = []
        self.music_files = []
        
        for file_path in self.audio_files:
            path_str = str(file_path).lower()
            if any(keyword in path_str for keyword in ['speech', 'voice', 'spoken', 'librispeech', 'commonvoice']):
                self.speech_files.append(file_path)
            else:
                self.music_files.append(file_path)
        
        # Calculate sampling probabilities
        n_speech = len(self.speech_files)
        n_music = len(self.music_files)
        
        # Apply temperature sampling
        speech_weight = (n_speech ** self.temperature_alpha)
        music_weight = (n_music ** self.temperature_alpha)
        total_weight = speech_weight + music_weight
        
        self.speech_prob = speech_weight / total_weight
        self.music_prob = music_weight / total_weight
        
        # Stage 3 music boost
        if self.stage == 3:
            music_boost = 3.0
            adjusted_music_prob = self.music_prob * music_boost
            total_adjusted = self.speech_prob + adjusted_music_prob
            self.speech_prob = self.speech_prob / total_adjusted
            self.music_prob = adjusted_music_prob / total_adjusted
        
        print(f"Sampling probabilities - Speech: {self.speech_prob:.3f}, Music: {self.music_prob:.3f}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def load_audio(self, file_path: Path) -> torch.Tensor:
        """Load and preprocess audio file"""
        try:
            waveform, sr = torchaudio.load(file_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            return waveform.squeeze(0)  # Remove channel dimension
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return silence as fallback
            return torch.zeros(self.sequence_samples)
    
    def crop_or_pad_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Crop or pad audio to target length"""
        if len(waveform) >= self.sequence_samples:
            # Random crop
            start_idx = random.randint(0, len(waveform) - self.sequence_samples)
            return waveform[start_idx:start_idx + self.sequence_samples]
        else:
            # Pad with zeros
            padding = self.sequence_samples - len(waveform)
            return torch.nn.functional.pad(waveform, (0, padding))
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Temperature sampling to select file type
        if random.random() < self.speech_prob and len(self.speech_files) > 0:
            file_path = random.choice(self.speech_files)
            audio_type = 'speech'
        else:
            file_path = random.choice(self.music_files) if len(self.music_files) > 0 else self.audio_files[idx]
            audio_type = 'music'
        
        # Load and preprocess audio
        waveform = self.load_audio(file_path)
        waveform = self.crop_or_pad_audio(waveform)
        
        # Generate language ID for speech (simplified)
        language_id = random.randint(0, 146) if audio_type == 'speech' else 0
        
        return {
            'waveform': waveform,
            'audio_type': audio_type,
            'language_id': language_id,
            'file_path': str(file_path)
        }


class WebDatasetLoader:
    """WebDataset-based data loader for large-scale training"""
    
    def __init__(self,
                 urls: List[str],
                 batch_size: int,
                 sample_rate: int = 24000,
                 sequence_length: float = 5.0,
                 num_workers: int = 4,
                 prefetch_factor: int = 4):
        
        self.urls = urls
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.sequence_samples = int(sequence_length * sample_rate)
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
    
    def preprocess_sample(self, sample):
        """Preprocess a single sample from WebDataset"""
        try:
            # Decode audio
            if 'flac' in sample:
                audio_data = sample['flac']
            elif 'wav' in sample:
                audio_data = sample['wav']
            else:
                return None
            
            # Load with torchaudio
            waveform, sr = torchaudio.load(io.BytesIO(audio_data))
            
            # Convert to mono and resample
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            waveform = waveform.squeeze(0)
            
            # Crop or pad
            if len(waveform) >= self.sequence_samples:
                start_idx = random.randint(0, len(waveform) - self.sequence_samples)
                waveform = waveform[start_idx:start_idx + self.sequence_samples]
            else:
                padding = self.sequence_samples - len(waveform)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # Determine audio type from metadata
            audio_type = sample.get('audio_type', 'unknown')
            language_id = sample.get('language_id', 0)
            
            return {
                'waveform': waveform,
                'audio_type': audio_type,
                'language_id': language_id
            }
        
        except Exception as e:
            print(f"Error processing sample: {e}")
            return None
    
    def create_dataloader(self) -> DataLoader:
        """Create WebDataset-based DataLoader"""
        dataset = (
            wds.WebDataset(self.urls)
            .shuffle(1000)
            .map(self.preprocess_sample)
            .select(lambda x: x is not None)  # Filter failed samples
            .batched(self.batch_size)
        )
        
        return wds.WebLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None,  # Batching handled by WebDataset
            prefetch_factor=self.prefetch_factor
        )


def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    
    # Training data loader
    if config.get('use_webdataset', False):
        train_urls = config['train_urls']
        train_loader = WebDatasetLoader(
            urls=train_urls,
            batch_size=config['batch_size'],
            sequence_length=config['sequence_length'],
            num_workers=config['num_workers']
        ).create_dataloader()
    else:
        train_dataset = AudioDataset(
            data_paths=config['train_paths'],
            sequence_length=config['sequence_length'],
            temperature_alpha=config.get('temperature_alpha', 0.7),
            stage=config.get('stage', 1)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            drop_last=True
        )
    
    # Validation data loader
    val_dataset = AudioDataset(
        data_paths=config['val_paths'],
        sequence_length=config['sequence_length'],
        temperature_alpha=1.0,  # No temperature sampling for validation
        stage=1
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching"""
    
    waveforms = torch.stack([item['waveform'] for item in batch])
    audio_types = [item['audio_type'] for item in batch]
    language_ids = torch.tensor([item['language_id'] for item in batch])
    
    return {
        'waveforms': waveforms,
        'audio_types': audio_types,
        'language_ids': language_ids
    }