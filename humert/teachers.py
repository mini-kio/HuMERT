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
    """mHuBERT-based speech teacher using HF pipeline + optional KMeans clustering.
    Falls back to simple conv features if transformers not installed or disabled.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_clusters = config.speech_clusters
        self.use_mhubert = getattr(config, 'speech_use_mhubert', True)
        self.kmeans_path = getattr(config, 'speech_kmeans_centroids_path', '')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._pipe = None
        self._centroids = None
        # Fallback lightweight conv stack if pipeline unavailable
        self.fallback = nn.Sequential(
            nn.Conv1d(1, 128, 400, stride=160, padding=200),
            nn.GELU(),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(256, 256)

    def _load_pipeline(self):
        if self._pipe is not None:
            return
        if not self.use_mhubert:
            return
        try:
            from transformers import pipeline
            self._pipe = pipeline("feature-extraction", model="utter-project/mHuBERT-147", device=0 if torch.cuda.is_available() else -1)
        except Exception:
            self._pipe = None
        # Load KMeans centroids if provided
        if self.kmeans_path and self.kmeans_path.strip():
            try:
                arr = torch.load(self.kmeans_path, map_location='cpu')
                if isinstance(arr, dict) and 'centroids' in arr:
                    arr = arr['centroids']
                self._centroids = arr.float()  # [K, D]
            except Exception:
                self._centroids = None

    def _extract_mhubert(self, waveform: torch.Tensor) -> torch.Tensor:
        self._load_pipeline()
        if self._pipe is None:
            return None  # signal fallback
        feats_list = []
        for w in waveform:  # iterate batch
            w_np = w.cpu().numpy()
            feats = self._pipe(w_np, sampling_rate=self.config.input_sample_rate)
            if isinstance(feats, list):
                feats = feats[0]
            feats_t = torch.tensor(feats, device=waveform.device, dtype=torch.float32)
            feats_list.append(feats_t)
        # Pad to max length
        max_len = max(f.size(0) for f in feats_list)
        dim = feats_list[0].size(-1)
        out = waveform.new_zeros(len(feats_list), max_len, dim)
        for i, f in enumerate(feats_list):
            out[i, :f.size(0)] = f
        return out  # [B, T, D]

    def _cluster(self, features: torch.Tensor) -> torch.Tensor:
        if self._centroids is None:
            # simple argmax over linear projection to num_clusters
            proj = torch.randn(features.size(-1), self.num_clusters, device=features.device)
            logits = features @ proj
            return logits.argmax(dim=-1)
        # L2 distance to centroids
        # features: [B,T,D], centroids: [K,D]
        f2 = (features**2).sum(-1, keepdim=True)
        c2 = (self._centroids.to(features.device)**2).sum(-1)  # [K]
        dots = features @ self._centroids.to(features.device).t()  # [B,T,K]
        dists = f2 - 2*dots + c2
        return dists.argmin(dim=-1)

    def forward(self, waveform: torch.Tensor, language_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if waveform.dim() == 2:
            wf = waveform.unsqueeze(1)
        else:
            wf = waveform
        feats = self._extract_mhubert(wf.squeeze(1)) if self.use_mhubert else None
        if feats is None:
            # fallback conv features
            convf = self.fallback(wf)  # [B,256,T']
            feats = convf.transpose(1,2)
        tokens = self._cluster(feats)
        return {
            'speech_tokens': tokens,
            'speech_features': feats
        }


class MusicalTeacher(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.cqt_bins = config.cqt_bins
        self.chroma_bins = config.chroma_bins
        self.hop_length = getattr(config, 'cqt_hop_length', 512)
        self.hop_length2 = getattr(config, 'cqt_hop_length2', 1024)
        self.sample_rate = 24000
        self.fmin = getattr(config, 'cqt_fmin', 32.7)
        self.multi_scale = getattr(config, 'music_multi_scale', False)

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
            cqt = torchaudio.functional.compute_cqt(
                waveform.to(torch.float32),
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmin=float(self.fmin),
                n_bins=self.cqt_bins,
            )
            cqt = torch.abs(cqt)
            cqt = torch.log(cqt + 1e-8)
            return cqt.transpose(1, 2)
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
            cqt = self.extract_cqt_features(waveform)
            if cqt.size(-1) % self.chroma_bins == 0:
                group = cqt.size(-1) // self.chroma_bins
                chroma = cqt.view(cqt.size(0), cqt.size(1), self.chroma_bins, group).mean(-1)
            else:
                chroma = F.interpolate(cqt.transpose(1, 2), size=self.chroma_bins, mode='linear', align_corners=False).transpose(1, 2)
            return chroma
        except Exception:
            feats = []
            for w in waveform:
                chroma = librosa.feature.chroma_cqt(y=w.cpu().numpy(), sr=self.sample_rate, hop_length=self.hop_length,
                                                    n_chroma=self.chroma_bins)
                feats.append(torch.tensor(chroma.T, device=waveform.device))
            return torch.stack(feats, dim=0)

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Base scale
        cqt_scale1 = self.extract_cqt_features(waveform)  # [B,T1,F]
        processed_cqt_scale1 = self.cqt_processor(cqt_scale1)
        processed_cqt_scale2 = None
        combined_for_music = processed_cqt_scale1

        if self.multi_scale and self.hop_length2 != self.hop_length:
            # Second scale with different hop
            orig_hop = self.hop_length
            self.hop_length = self.hop_length2
            cqt_scale2 = self.extract_cqt_features(waveform)  # [B,T2,F]
            self.hop_length = orig_hop
            # Time align scale2 to scale1 length (simple linear interp)
            if cqt_scale2.size(1) != cqt_scale1.size(1):
                target_len = max(cqt_scale1.size(1), cqt_scale2.size(1))
                if cqt_scale1.size(1) != target_len:
                    cqt_scale1 = F.interpolate(cqt_scale1.transpose(1,2), size=target_len, mode='linear', align_corners=False).transpose(1,2)
                    processed_cqt_scale1 = F.interpolate(processed_cqt_scale1.transpose(1,2), size=target_len, mode='linear', align_corners=False).transpose(1,2)
                if cqt_scale2.size(1) != target_len:
                    cqt_scale2 = F.interpolate(cqt_scale2.transpose(1,2), size=target_len, mode='linear', align_corners=False).transpose(1,2)
            processed_cqt_scale2 = self.cqt_processor(cqt_scale2)
            # Combine (concat then reduce) to produce final music feature base
            combined_cat = torch.cat([processed_cqt_scale1, processed_cqt_scale2], dim=-1)
            if not hasattr(self, 'multi_reduce') or self.multi_reduce.in_features != combined_cat.size(-1):
                self.multi_reduce = nn.Linear(combined_cat.size(-1), self.cqt_bins).to(combined_cat.device)
            combined_for_music = self.multi_reduce(combined_cat)

        chroma_features = self.extract_chroma_features(waveform)
        processed_cqt = combined_for_music  # rename for downstream compatibility
        processed_chroma = self.chroma_processor(chroma_features)
        combined = torch.cat([processed_cqt, processed_chroma], dim=-1)
        output_features = self.output_proj(combined)
        out: Dict[str, torch.Tensor] = {
            'music_features': output_features,
            'cqt_features': processed_cqt,
            'chroma_features': chroma_features,
        }
        if processed_cqt_scale2 is not None:
            out['cqt_features_scale1'] = processed_cqt_scale1
            out['cqt_features_scale2'] = processed_cqt_scale2
        return out