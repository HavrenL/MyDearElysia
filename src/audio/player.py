"""
音频播放模块
"""
import io
import time

import sounddevice as sd
import soundfile as sf

from src.utils.logger import get_logger

log = get_logger("player")


def play_audio(audio_bytes: bytes) -> None:
    """播放音频字节流"""
    with io.BytesIO(audio_bytes) as f:
        data, samplerate = sf.read(f)
        duration = len(data) / samplerate
        log.info(f"播放语音: {duration:.1f}s @ {samplerate}Hz")
        sd.play(data, samplerate)
        sd.wait()
    log.info("播放完成")


def play_audio_data(data, samplerate: int) -> None:
    """直接播放音频数据"""
    duration = len(data) / samplerate
    log.info(f"播放语音: {duration:.1f}s @ {samplerate}Hz")
    sd.play(data, samplerate)
    sd.wait()
    log.info("播放完成")
