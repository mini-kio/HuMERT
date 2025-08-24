import argparse
import os, sys
import torch
import torchaudio
from typing import Tuple

# Ensure project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.model_config import get_model_config
from humert.teachers import DACTeacher

def chunk_indices(total:int, chunk:int, overlap:int):
    if chunk <= overlap:
        raise ValueError('chunk must be > overlap')
    start = 0
    while start < total:
        end = min(total, start + chunk)
        yield start, end
        if end == total:
            break
        start = start + (chunk - overlap)

def build_window(length:int):
    # Hann window for smooth crossfade
    return torch.hann_window(length, periodic=False)

def encode_decode_chunk(model:DACTeacher, wav:torch.Tensor, sr:int) -> torch.Tensor:
    # wav: [T] mono float -1..1
    device = model.device
    x = wav.to(device).unsqueeze(0).unsqueeze(0)  # [B,1,T]
    # Preprocess expects same sample rate as model.sample_rate
    x_p = model.dac_model.preprocess(x, sr)
    z, codes, latents, *rest = model.dac_model.encode(x_p)
    # Sequential fallback decoding attempts
    attempts = []
    def _finalize(t):
        if isinstance(t, dict):
            # common key guesses
            for k in ['audio','wav','waveform']:
                if k in t:
                    t = t[k]
                    break
        if isinstance(t, (list, tuple)):
            t = t[0]
        if t.dim() == 3:
            t = t.squeeze(0).squeeze(0)
        elif t.dim() == 2:
            t = t.squeeze(0)
        return t.detach().cpu()
    # 1) decode(codes)
    try:
        recon = model.dac_model.decode(codes)
        return _finalize(recon)
    except Exception as e:
        attempts.append(f'decode(codes): {e}')
    # 2) decode(z)
    try:
        recon = model.dac_model.decode(z)
        return _finalize(recon)
    except Exception as e:
        attempts.append(f'decode(z): {e}')
    # 3) direct decoder(z)
    if hasattr(model.dac_model, 'decoder'):
        try:
            recon = model.dac_model.decoder(z)
            return _finalize(recon)
        except Exception as e:
            attempts.append(f'decoder(z): {e}')
    # 4) if latents available try decode(latents)
    try:
        recon = model.dac_model.decode(latents)
        return _finalize(recon)
    except Exception as e:
        attempts.append(f'decode(latents): {e}')
    # Failure
    raise RuntimeError('All decode attempts failed:\n' + '\n'.join(attempts) + f"\nAvailable attrs: {[a for a in dir(model.dac_model) if 'dec' in a.lower()]}")

def main():
    ap = argparse.ArgumentParser(description='DAC round-trip (chunked overlap-add)')
    ap.add_argument('--input', type=str, default='you and me.mp3')
    ap.add_argument('--output', type=str, default='reconstructed.wav')
    ap.add_argument('--chunk_sec', type=float, default=2.0)
    ap.add_argument('--overlap_sec', type=float, default=0.2)
    ap.add_argument('--model_type', type=str, default='24khz')
    ap.add_argument('--bitrate', type=str, default='8kbps')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    config = get_model_config()
    teacher = DACTeacher(config, model_type=args.model_type, model_bitrate=args.bitrate, lazy_load=False, device=args.device)
    sr_target = teacher.dac_sample_rate or config.input_sample_rate

    ext = os.path.splitext(args.input)[1].lower()
    try:
        wav, sr = torchaudio.load(args.input)
    except Exception:
        if ext in ['.mp4', '.m4a', '.aac']:
            # ffmpeg fallback -> pcm16 wav bytes
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            cmd = [
                'ffmpeg','-y','-i', args.input,
                '-vn','-ac','1','-ar', str(sr_target), '-f','wav', tmp_path
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                wav, sr = torchaudio.load(tmp_path)
            finally:
                try: os.remove(tmp_path)
                except OSError: pass
        else:
            raise
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != sr_target:
        wav = torchaudio.transforms.Resample(sr, sr_target)(wav)
        sr = sr_target
    wav = wav.squeeze(0)

    chunk_samples = int(args.chunk_sec * sr)
    overlap_samples = int(args.overlap_sec * sr)

    out = torch.zeros_like(wav)
    weight = torch.zeros_like(wav)

    win_cache = {}

    for s,e in chunk_indices(len(wav), chunk_samples, overlap_samples):
        chunk = wav[s:e]
        recon = encode_decode_chunk(teacher, chunk, sr)
        # If recon length differs (codec latency), center crop or pad
        if recon.size(0) > chunk.size(0):
            recon = recon[:chunk.size(0)]
        elif recon.size(0) < chunk.size(0):
            recon = torch.nn.functional.pad(recon, (0, chunk.size(0)-recon.size(0)))
        L = recon.size(0)
        if L not in win_cache:
            win_cache[L] = build_window(L)
        w = win_cache[L]
        out[s:s+L] += recon * w
        weight[s:s+L] += w

    weight = torch.where(weight==0, torch.ones_like(weight), weight)
    out = out / weight
    # Normalize
    peak = out.abs().max().clamp(min=1e-6)
    out = (out / peak * 0.95).unsqueeze(0)
    torchaudio.save(args.output, out, sr)
    print(f'Saved reconstructed audio -> {args.output} (sr={sr})')

if __name__ == '__main__':
    main()
