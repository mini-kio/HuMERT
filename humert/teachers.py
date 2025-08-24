"""Multi-Teacher System: DAC, Speech, Musical Teachers (clean implementation)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Optional, Tuple, Dict, Any
from config.model_config import ModelConfig


class DACTeacher(nn.Module):
    """Wrapper around Descript Audio Codec with caching & optional resample/upmix."""

    def __init__(
        self,
        config: ModelConfig,
        model_type: str = "24khz",
        model_bitrate: str = "8kbps",
        tag: str = "latest",
        device: Optional[str] = None,
        lazy_load: bool = True,
        max_cache_items: int = 64,
    ) -> None:
        super().__init__()
        self.config = config
        self.model_type = model_type
        self.model_bitrate = model_bitrate
        self.tag = tag
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lazy_load = lazy_load
        self.max_cache_items = max_cache_items
        self.dac_model: Optional[nn.Module] = None
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.num_codebooks = config.dac_codebooks
        self.vocab_size = config.dac_vocab_size
        self.resampler: Optional[nn.Module] = None
        self.target_channels = getattr(config, "dac_target_channels", 2)
        self.input_sample_rate = config.input_sample_rate
        self.dac_sample_rate: Optional[int] = None
        if not self.lazy_load:
            self._ensure_model_loaded()

    def _add_local_dac_to_path(self):
        import sys, os
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        local_dac_parent = os.path.join(root_dir, 'Dac')
        if os.path.isdir(local_dac_parent) and local_dac_parent not in sys.path:
            sys.path.append(local_dac_parent)

    def _load_dac(self):
        self._add_local_dac_to_path()
        try:
            import dac  # type: ignore
        except ImportError as e:
            raise ImportError("dac 패키지 불러오기 실패 (audiotools 설치 필요)") from e
        try:
            model_path = dac.utils.download(model_type=self.model_type, model_bitrate=self.model_bitrate, tag=self.tag)
            model = dac.DAC.load(model_path)
        except Exception as e:
            raise RuntimeError(f'DAC 모델 로드 실패: {e}')
        model = model.to(self.device).eval()
        return model

    def _ensure_model_loaded(self):
        if self.dac_model is None:
            self.dac_model = self._load_dac()
            self.dac_sample_rate = getattr(self.dac_model, 'sample_rate', None)
            # If loaded DAC is mono (most 24k models), override target_channels to 1
            expected_ch = getattr(getattr(self.dac_model, 'encoder', None), 'in_channels', 1)
            if (self.model_type.startswith('24') or (self.dac_sample_rate and self.dac_sample_rate <= 26000)) and expected_ch == 1:
                self.target_channels = 1
            try:
                import torchaudio
                if self.dac_sample_rate and self.dac_sample_rate != self.input_sample_rate:
                    self.resampler = torchaudio.transforms.Resample(
                        orig_freq=self.input_sample_rate,
                        new_freq=self.dac_sample_rate
                    ).to(self.device)
            except Exception:
                self.resampler = None

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._ensure_model_loaded()
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # [B,1,T]
        waveform = waveform.to(self.device)
        # Upmix only if model expects >1 channels (e.g., 44.1k stereo); avoid for 24k mono models
        if self.target_channels > 1 and waveform.size(1) == 1:
            if self.dac_sample_rate and self.dac_sample_rate > 30000:
                waveform = waveform.repeat(1, self.target_channels, 1)
        # Resample 24k -> 44.1k if required
        if self.resampler is not None:
            # torchaudio resampler expects [B, C, T]
            B, C, T = waveform.shape
            waveform = self.resampler(waveform.view(B * C, 1, T)).view(B, C, -1)
        sig = waveform.float()
        batch_hash = hash((int(sig.sum().item()*1e6), sig.shape[-1], sig.shape[0]))
        if batch_hash in self._cache:
            cached = self._cache[batch_hash]
            return {k: v.clone() for k, v in cached.items()}
        x = self.dac_model.preprocess(waveform, self.dac_model.sample_rate)
        z, codes, latents, _, _ = self.dac_model.encode(x)
        codes_reordered = codes.permute(0, 2, 1).contiguous()
        if codes_reordered.size(-1) != self.num_codebooks:
            self.num_codebooks = codes_reordered.size(-1)
        outputs = {
            'dac_tokens': codes_reordered,
            'dac_latents': z.transpose(1, 2).contiguous(),
            'dac_sample_rate': torch.tensor(self.dac_sample_rate or 0),
            'input_sample_rate': torch.tensor(self.input_sample_rate),
            'dac_channels_used': torch.tensor(waveform.size(1)),
        }
        if len(self._cache) >= self.max_cache_items:
            self._cache.pop(next(iter(self._cache)))
        self._cache[batch_hash] = {k: v.detach().clone() for k, v in outputs.items()}
        return outputs


class SpeechTeacher(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_clusters = config.speech_clusters
        self.num_languages = config.speech_languages

        self.mel_extractor = nn.Conv1d(1, 80, 400, stride=160, padding=200)
        self.mfcc_extractor = nn.Conv1d(80, 13, 1)

        self.feature_proj = nn.Linear(93, 256)
        self.lang_proj = nn.Linear(256 + 64, 256)
        self.cluster_proj = nn.Linear(256, self.num_clusters)

        self.language_embedding = nn.Embedding(self.num_languages, 64)
        self.temperature_alpha = 0.7

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        mel = self.mel_extractor(waveform)
        mel = torch.log(mel.clamp(min=1e-8))
        mfcc = self.mfcc_extractor(mel)
        combined = torch.cat([mel, mfcc], dim=1)
        return combined.transpose(1, 2)

    def cluster_features(self, features: torch.Tensor, language_ids: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_proj(features)
        if language_ids is not None:
            lang_emb = self.language_embedding(language_ids).unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, lang_emb], dim=-1)
            x = self.lang_proj(x)
        logits = self.cluster_proj(x)
        probs = F.softmax(logits / self.temperature_alpha, dim=-1)
        return logits, probs

    def forward(self, waveform: torch.Tensor, language_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        features = self.extract_features(waveform)
        logits, probs = self.cluster_features(features, language_ids)
        tokens = logits.argmax(dim=-1)
        return {
            'speech_tokens': tokens,
            'speech_probs': probs,
            'speech_features': features
        }


class MusicalTeacher(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.cqt_bins = config.cqt_bins
        self.chroma_bins = config.chroma_bins
        self.hop_length = getattr(config, 'cqt_hop_length', 512)
        self.sample_rate = 24000
        self.fmin = getattr(config, 'cqt_fmin', 32.7)

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
        self.output_proj = nn.Linear(self.cqt_bins + self.chroma_bins, self.cqt_bins)

    def extract_cqt_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Batch CQT extraction with torchaudio (fallback to librosa) -> [B, T, cqt_bins]."""
        try:
            import torchaudio
            # torchaudio cqt: output shape [B, freq, time]
            cqt = torchaudio.functional.compute_cqt(
                waveform.to(torch.float32),
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmin=float(self.fmin),  # configurable
                n_bins=self.cqt_bins,
            )  # [B, freq, time]
            cqt = torch.abs(cqt)
            cqt = torch.log(cqt + 1e-8)
            return cqt.transpose(1, 2)  # [B, time, freq]
        except Exception:
            feats = []
            for w in waveform:
                cqt = librosa.cqt(w.cpu().numpy(), sr=self.sample_rate, hop_length=self.hop_length,
                                  n_bins=self.cqt_bins, bins_per_octave=12)
                cqt_log = np.log(np.abs(cqt) + 1e-8)
                feats.append(torch.tensor(cqt_log.T, device=waveform.device))
            return torch.stack(feats, dim=0)

    def extract_chroma_features(self, waveform: torch.Tensor) -> torch.Tensor:
        try:
            import torchaudio
            # Approximate chroma via projecting CQT bins into 12 classes (simple pooling)
            cqt = self.extract_cqt_features(waveform)  # [B,T,F]
            if cqt.size(-1) % self.chroma_bins == 0:
                group = cqt.size(-1) // self.chroma_bins
                chroma = cqt.view(cqt.size(0), cqt.size(1), self.chroma_bins, group).mean(-1)
            else:
                chroma = F.interpolate(cqt.transpose(1,2), size=self.chroma_bins, mode='linear', align_corners=False).transpose(1,2)
            return chroma
        except Exception:
            feats = []
            for w in waveform:
                chroma = librosa.feature.chroma_cqt(y=w.cpu().numpy(), sr=self.sample_rate, hop_length=self.hop_length,
                                                    n_chroma=self.chroma_bins)
                feats.append(torch.tensor(chroma.T, device=waveform.device))
            return torch.stack(feats, dim=0)

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        cqt_features = self.extract_cqt_features(waveform)
        chroma_features = self.extract_chroma_features(waveform)
        processed_cqt = self.cqt_processor(cqt_features)
        processed_chroma = self.chroma_processor(chroma_features)
        combined = torch.cat([processed_cqt, processed_chroma], dim=-1)
        output_features = self.output_proj(combined)
        return {
            'music_features': output_features,
            'cqt_features': cqt_features,
            'chroma_features': chroma_features
        }