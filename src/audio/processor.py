"""
音频预处理模块
"""
import numpy as np
import noisereduce as nr
import pyloudnorm as pyln
from src.config.settings import settings
from src.utils.logger import logger


class AudioProcessor:
    """音频预处理：降噪 + 响度标准化"""

    def __init__(self, sample_rate: int = None, loudness_target: float = None):
        cfg = settings.get("audio", {})
        self.sample_rate = sample_rate or cfg.get("sample_rate", 16000)
        self.loudness_target = loudness_target or cfg.get("loudness_target", -20.0)
        self.meter = pyln.Meter(self.sample_rate)

    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """完整的音频预处理流水线"""
        noise_clip = self._extract_noise_sample(audio_data)
        reduced_noise = self._spectral_reduction(audio_data, noise_clip)
        normalized = self._loudness_normalization(reduced_noise)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _extract_noise_sample(self, audio_data: np.ndarray) -> np.ndarray:
        """提取噪声样本"""
        cfg = settings.get("audio", {})
        duration = cfg.get("noise_sample_duration", 0.2)
        ratio = cfg.get("noise_sample_ratio", 0.5)
        if len(audio_data) > self.sample_rate * ratio:
            return audio_data[:int(self.sample_rate * duration)]
        return audio_data.copy()

    def _spectral_reduction(self, audio_data: np.ndarray, noise_clip: np.ndarray) -> np.ndarray:
        """谱减法降噪"""
        cfg = settings.get("audio", {})
        return nr.reduce_noise(
            sr=self.sample_rate,
            y=audio_data.astype(np.float32),
            y_noise=noise_clip.astype(np.float32),
            stationary=cfg.get("denoise_stationary", True),
            prop_decrease=cfg.get("denoise_strength", 0.9)
        )

    def _loudness_normalization(self, audio_data: np.ndarray) -> np.ndarray:
        """响度标准化"""
        current_loudness = self.meter.integrated_loudness(audio_data.reshape(-1, 1))
        return pyln.normalize.loudness(
            audio_data.astype(np.float32),
            current_loudness,
            self.loudness_target
        )
