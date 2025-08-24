import torch, os, sys

# Ensure project root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from humert.teachers import DACTeacher
from config.model_config import get_model_config
import torchaudio

AUDIO_PATH = 'you and me.mp3'
assert os.path.exists(AUDIO_PATH), f'Missing audio file {AUDIO_PATH}'

config = get_model_config()
# Force 24k model usage if available
teacher = DACTeacher(config, model_type='24khz', model_bitrate='8kbps', lazy_load=False)

try:
    waveform, sr = torchaudio.load(AUDIO_PATH)
except Exception as e:
    raise RuntimeError(f"오디오 로드 실패: {e}")
if sr != config.input_sample_rate:
    waveform = torchaudio.transforms.Resample(sr, config.input_sample_rate)(waveform)
# mono mix
if waveform.shape[0] > 1:
    waveform = waveform.mean(0, keepdim=True)
waveform = waveform[:, :config.input_sample_rate * 2]  # max 2s clip
waveform = waveform.squeeze(0).unsqueeze(0)  # [B,T]

out = teacher(waveform)
print('DAC keys:', list(out.keys()))
print('dac_tokens shape', out['dac_tokens'].shape)
print('dac_latents shape', out['dac_latents'].shape)
print('sample rates', out['input_sample_rate'], out['dac_sample_rate'])
print('unique first codebook tokens', torch.unique(out['dac_tokens'][:, :, 0]).numel())
