# 🚀 HuMERT-300M: Universal Audio Understanding

HuMERT-300M is a cutting-edge universal audio understanding model that combines Flash Linear Transformer architecture with multi-teacher learning for both speech and music tasks.

## 🏗️ Architecture Overview

- **Parameters**: 300M (optimized from 350M)
- **Encoder**: 22-layer Flash Linear Transformer with FAVOR+ attention
- **Training**: 2-stage curriculum learning (15.5 days on 2x RTX 3090)
- **Tasks**: Speech recognition, Music understanding, Audio reconstruction

### Core Components

1. **Flash Linear Transformer Encoder** (O(n) complexity)
   - 22 layers, 1024 hidden dimension, 16 attention heads
   - FAVOR+ attention with ALiBi positional bias
   - Adapter gating for task-specific adaptation

2. **ConvFrontend24k** 
   - Raw 24kHz waveform input processing
   - 320x downsampling with convolutional blocks
   - Integrated masking for self-supervised learning

3. **Multi-Teacher System**
   - **DAC Teacher**: 9 codebooks for acoustic reconstruction  
   - **Speech Teacher**: 1500 clusters with multilingual support
   - **Musical Teacher**: 84-bin CQT + 12-bin chroma features

4. **Prediction Heads**
   - Adapter-based architecture (64-dim adapters)
   - DAC: 9216-dim output (9 × 1024 codebooks)
   - Speech: 1500-dim cluster prediction
   - Music: 84-dim continuous features

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-repo/HuMERT.git
cd HuMERT
pip install -r requirements.txt
```

### Training

1. **Prepare your data configuration**:
```bash
cp config/train_config.json config/my_train_config.json
# Edit paths to your speech and music datasets
```

2. **Start training**:
```bash
python scripts/train.py --config config/my_train_config.json --output_dir ./outputs
```

3. **Resume from checkpoint**:
```bash
python scripts/train.py --config config/my_train_config.json --resume ./outputs/checkpoints/checkpoint_stage_2_step_150000.pt
```

### Inference

```python
import torch
from humert import HuMERTModel
from config.model_config import get_model_config

# Load model
config = get_model_config()
model = HuMERTModel(config)
model.load_state_dict(torch.load('humert_300m_final.pt')['model_state_dict'])

# Extract features
waveform = torch.randn(1, 24000 * 5)  # 5 seconds at 24kHz
features = model.extract_features(waveform)  # [1, 375, 1024]

# Task-specific inference
outputs = model.inference(waveform, task='speech')  # or 'music', 'dac', 'all'
```

## 📊 Training Stages

### Stage 1: Warm-up (60K steps, 5 seconds)
- **Purpose**: Mask learning stabilization + teacher-student alignment
- **Batch Size**: 12 per GPU × 2 GPUs × 8 accumulation = 192 effective
- **Memory**: <15GB per GPU
- **Duration**: ~4.4 days

### Stage 2: Main Training (240K steps, 5 seconds) 
- **Purpose**: Full loss convergence + GradNorm scaling
- **Batch Size**: 192 effective (same as Stage 1)
- **Memory**: <17GB per GPU  
- **Duration**: Continues from Stage 1

### Stage 3: Long-range Spike (35K steps, 8 seconds)
- **Purpose**: Musical structure + long-context learning
- **Batch Size**: 6 per GPU × 2 GPUs × 16 accumulation = 192 effective
- **Optimizations**: Freeze layers 0-15, music:speech = 3:1 sampling
- **Memory**: <19GB per GPU
- **Duration**: ~0.75 days

**Total Training Time**: 15.5 days wall-clock time

## 🔧 Memory Optimizations

- **Flash Attention 2.5+**: Required for efficient O(n) attention
- **ZeRO-2**: Distributed optimizer state sharding
- **Activation Checkpointing**: Gradient checkpointing enabled
- **Mixed Precision**: FP16 training with FP8 optimizer states
- **Stage 3 Adapter Freezing**: 1.2GB memory savings

## 📈 Performance Targets

- **ML-SUPERB**: +0.6~1.3 improvement across all tasks
- **MERT Tasks**: +1.4 improvement on 14 music tasks  
- **Training Efficiency**: 2.3x faster than baseline
- **Memory Usage**: <17GB Stage 1-2, <19GB Stage 3

## 🛠️ Development

### Project Structure
```
HuMERT/
├── humert/              # Core model implementation
│   ├── model.py         # Main HuMERT model
│   ├── encoder.py       # Flash Linear Transformer
│   ├── frontend.py      # Audio preprocessing
│   ├── teachers.py      # Multi-teacher system
│   └── heads.py         # Prediction heads
├── config/              # Configuration files
├── training/            # Training utilities
├── scripts/             # Training and evaluation scripts
└── data/               # Data loading and processing
```

### Key Dependencies
- `torch>=2.0.0` with `flash-attn>=2.5.0`
- `bitsandbytes>=0.41.0` for 8-bit optimization
- `librosa>=0.10.0` for audio processing
- `wandb>=0.15.0` for experiment tracking

## 📝 Configuration

### Model Configuration
```python
# config/model_config.py
ModelConfig(
    num_layers=22,           # Encoder layers
    hidden_dim=1024,         # Hidden dimension
    num_attention_heads=16,  # Attention heads
    dac_codebooks=9,         # DAC codebooks (increased from 8)
    speech_clusters=1500,    # Speech clusters (reduced from 2000)
    cqt_bins=84             # Musical CQT bins
)
```

### Training Configuration  
```python
# config/model_config.py
TrainingConfig(
    learning_rate=1e-4,
    gradient_clipping=1.0,
    stage1_steps=60000,      # Warm-up
    stage2_steps=240000,     # Main training
    stage3_steps=35000,      # Long-range spike
    gradnorm_alpha=1.5       # GradNorm weighting
)
```

## 🎯 Benchmarks & Evaluation

The model targets state-of-the-art performance on:
- **ML-SUPERB**: 10 speech understanding tasks
- **MERT Benchmarks**: 14 music understanding tasks  
- **Audio Quality**: DAC reconstruction with 9 codebooks
- **Efficiency**: 2.3x training speedup vs baseline models


## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgments

- Flash Attention implementation based on [Dao et al. 2022](https://arxiv.org/abs/2205.14135)
- FAVOR+ attention from [Choromanski et al. 2021](https://arxiv.org/abs/2009.14794)
- DAC compression from [Kumar et al. 2023](https://arxiv.org/abs/2306.06546)
- Multi-teacher learning inspired by [Chen et al. 2022](https://arxiv.org/abs/2206.04669)


**HuMERT-300M**: Universal Audio Understanding in 300M Parameters 🎵🎤