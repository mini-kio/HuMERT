import torch, torchaudio, os
from humert.teachers import DACTeacher
from config.model_config import get_model_config

def main():
    config = get_model_config()
    teacher = DACTeacher(config, model_type='24khz', model_bitrate='8kbps', lazy_load=False)
    print('Loaded DAC sample rate:', teacher.dac_sample_rate)
    audio = 'you and me.mp3'
    if not os.path.isfile(audio):
        raise FileNotFoundError('Audio file you and me.mp3 not found in workspace root')
    wav, sr = torchaudio.load(audio)
    print('Original sr:', sr, 'shape:', wav.shape)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != config.input_sample_rate:
        wav = torchaudio.transforms.Resample(sr, config.input_sample_rate)(wav)
    wav = wav.squeeze(0).unsqueeze(0)  # [B,T]
    with torch.no_grad():
        out = teacher(wav)
    print('dac_tokens', out['dac_tokens'].shape, 'dac_latents', out['dac_latents'].shape)
    print('meta sample rates', out['input_sample_rate'].item(), out['dac_sample_rate'].item())

if __name__ == '__main__':
    main()