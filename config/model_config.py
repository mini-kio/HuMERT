"""
HuMERT-300M Model Configuration
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    # Flash Linear Transformer Encoder
    num_layers: int = 22
    hidden_dim: int = 1024
    num_attention_heads: int = 16
    feedforward_dim: int = 4096
    dropout: float = 0.1
    
    # ConvFrontend24k
    input_sample_rate: int = 24000
    conv_channels: int = 512
    conv_kernel_size: int = 10
    conv_stride: int = 5
    conv_blocks: int = 4
    conv_block_strides: list = None
    downsampling_factor: int = 320
    
    # Masking
    mask_prob: float = 0.65
    mask_length: int = 10
    
    # Adapter
    adapter_dim: int = 64
    
    # DAC Teacher
    dac_codebooks: int = 9
    dac_frame_rate: int = 50
    dac_vocab_size: int = 1024
    dac_model_type: str = '44khz'        # actual pretrained DAC model sample rate
    dac_model_bitrate: str = '8kbps'
    dac_target_channels: int = 2         # stereo expected by 44.1k model; we upmix if mono
    
    # Speech Teacher
    speech_clusters: int = 1500
    speech_languages: int = 147
    
    # Musical Teacher
    cqt_bins: int = 84
    chroma_bins: int = 12
    cqt_fmin: float = 32.7
    cqt_hop_length: int = 512

    # Alignment options
    alignment_return_indices: bool = False  # if True, return mapping indices for debugging
    alignment_smooth_kernel: int = 0        # odd >1 applies temporal smoothing after alignment (continuous feats)
    
    # Logging / diagnostics
    log_codebook_stats: bool = True         # log codebook perplexity & unique token counts during training
    log_codebook_moving_avg: bool = True    # keep moving average of perplexities
    codebook_mavg_alpha: float = 0.01       # EMA factor for moving averages

    # Alignment refinement
    alignment_linear_interpolate: bool = True  # use linear interpolation for continuous teacher features

    # Dynamic masking
    mask_prob_schedule: tuple = None        # e.g., ((0,0.65),(100000,0.55),(200000,0.5))
    mask_length_range: tuple = (6, 14)      # random uniform length in range

    # Contrastive regularization
    use_contrastive: bool = True
    contrastive_dim: int = 256
    contrastive_temperature: float = 0.1
    contrastive_subsample: int = 4096        # max samples for contrastive per batch
    multi_scale_consistency_weight: float = 0.1

    # Music multi-scale
    music_multi_scale: bool = True
    cqt_hop_length2: int = 1024             # second scale hop length

    # Speech teacher (external mHuBERT pipeline & optional KMeans centroids file)
    speech_use_mhubert: bool = True
    speech_kmeans_centroids_path: str = ''
    
    def __post_init__(self):
        if self.conv_block_strides is None:
            self.conv_block_strides = [2, 2, 2, 2]


@dataclass
class TrainingConfig:
    # Optimizer
    learning_rate: float = 1e-4
    betas: tuple = (0.9, 0.98)
    weight_decay: float = 0.02
    gradient_clipping: float = 1.0
    warmup_steps: int = 8000
    
    # Training stages
    stage1_steps: int = 60000
    stage1_sequence_length: int = 5
    stage1_batch_size: int = 12
    
    stage2_steps: int = 240000
    stage2_sequence_length: int = 5
    stage2_batch_size: int = 12
    
    stage3_steps: int = 35000
    stage3_sequence_length: int = 8
    stage3_batch_size: int = 6
    
    gradient_accumulation: int = 8
    num_gpus: int = 2
    
    # Loss weights
    gradnorm_alpha: float = 1.5
    
    # Memory optimization
    use_flash_attention: bool = True
    use_activation_checkpointing: bool = True
    mixed_precision: bool = True
    
    # Data
    temperature_sampling_alpha: float = 0.7
    stage3_music_boost_ratio: float = 3.0
    
    # Validation
    val_frequency: int = 5000
    val_patience: int = 3

    # Scheduler / total steps
    total_steps: int = 0  # If 0, will be computed from stage steps
    min_lr_ratio: float = 0.1  # floor_lr = base_lr * min_lr_ratio

    # EMA (Exponential Moving Average)
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_update_after: int = 1000            # start EMA updates after this many global steps
    ema_device: str = 'cpu'                 # store EMA weights on cpu to save GPU memory
    use_ema_for_eval: bool = True           # swap to EMA weights during validation
    
    # Contrastive loss weight (scales inside GradNorm still)
    contrastive_loss_scale: float = 1.0

    def __post_init__(self):
        if self.total_steps == 0:
            self.total_steps = self.stage1_steps + self.stage2_steps + self.stage3_steps


def get_model_config() -> ModelConfig:
    return ModelConfig()


def get_training_config() -> TrainingConfig:
    return TrainingConfig()