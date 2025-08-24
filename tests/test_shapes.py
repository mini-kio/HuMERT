import torch
import sys, os

# Ensure project root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from humert.model import HuMERTModel
from config.model_config import get_model_config


def test_forward_shapes():
    config = get_model_config()
    model = HuMERTModel(config)
    model.dac_teacher.lazy_load = True  # avoid immediate download until first call
    B = 2
    samples = 24000 * 1  # 1s audio
    waveform = torch.randn(B, samples)
    out = model(waveform, return_teacher_labels=False)
    assert 'encoded_features' in out
    enc = out['encoded_features']
    assert enc.dim() == 3
    # prediction heads
    assert 'dac_logits' in out and 'speech_logits' in out and 'music_features' in out
    assert out['dac_logits'].shape[0] == B
    assert out['speech_logits'].shape[0] == B
    assert out['music_features'].shape[0] == B


def test_teacher_alignment_dummy():
    # Use smaller config to speed
    config = get_model_config()
    model = HuMERTModel(config)
    model.dac_teacher.lazy_load = True
    B = 1
    waveform = torch.randn(B, 24000)
    # Monkeypatch teachers to return controlled sizes
    with torch.no_grad():
        model.dac_teacher.forward = lambda w: {'dac_tokens': torch.randint(0, config.dac_vocab_size, (B, 50, config.dac_codebooks), dtype=torch.long)}
        model.speech_teacher.forward = lambda w: {'speech_tokens': torch.randint(0, config.speech_clusters, (B, 60), dtype=torch.long)}
        model.musical_teacher.forward = lambda w: {'music_features': torch.randn(B, 40, config.cqt_bins)}
    out = model(waveform, return_teacher_labels=True)
    L = out['encoded_features'].size(1)
    assert out['dac_tokens'].size(1) == L
    assert out['speech_tokens'].size(1) == L
    assert out['music_features'].size(1) == L

def test_alignment_index_monotonic():
    config = get_model_config()
    model = HuMERTModel(config)
    model.dac_teacher.lazy_load = True
    B = 1
    # Create waveform 48k samples (2s at 24k) to test scaling
    waveform = torch.randn(B, 48000)
    # Controlled lengths
    Td = 73  # arbitrary
    Ts = 90
    Tm = 64
    with torch.no_grad():
        model.dac_teacher.forward = lambda w: {'dac_tokens': torch.randint(0, config.dac_vocab_size, (B, Td, config.dac_codebooks), dtype=torch.long)}
        model.speech_teacher.forward = lambda w: {'speech_tokens': torch.randint(0, config.speech_clusters, (B, Ts), dtype=torch.long)}
        model.musical_teacher.forward = lambda w: {'music_features': torch.randn(B, Tm, config.cqt_bins)}
    out = model(waveform, return_teacher_labels=True)
    L = out['encoded_features'].size(1)
    assert out['dac_tokens'].shape == (B, L, config.dac_codebooks)
    assert out['speech_tokens'].shape == (B, L)
    assert out['music_features'].shape == (B, L, config.cqt_bins)
    # Value range validity
    assert out['dac_tokens'].dtype == torch.long
    assert out['speech_tokens'].dtype == torch.long

def test_alignment_indices_and_smoothing():
    config = get_model_config()
    config.alignment_return_indices = True
    config.alignment_smooth_kernel = 3
    model = HuMERTModel(config)
    model.dac_teacher.lazy_load = True
    B = 1
    waveform = torch.randn(B, 36000)
    with torch.no_grad():
        model.dac_teacher.forward = lambda w: {'dac_tokens': torch.randint(0, config.dac_vocab_size, (B, 45, config.dac_codebooks), dtype=torch.long)}
        model.speech_teacher.forward = lambda w: {'speech_tokens': torch.randint(0, config.speech_clusters, (B, 55), dtype=torch.long)}
        model.musical_teacher.forward = lambda w: {'music_features': torch.randn(B, 30, config.cqt_bins)}
    out = model(waveform, return_teacher_labels=True)
    assert 'dac_frame_indices' in out and 'speech_frame_indices' in out and 'music_frame_indices' in out
    assert out['music_features'].shape[1] == out['dac_tokens'].shape[1]


if __name__ == '__main__':
    test_forward_shapes()
    test_teacher_alignment_dummy()
    test_alignment_index_monotonic()
    test_alignment_indices_and_smoothing()
    print('Shape tests passed')
