"""
语音识别模块 — 录音 + faster-whisper 转文字
"""
import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions

from src.audio.processor import AudioProcessor
from src.config.settings import settings
from src.utils.logger import logger, get_logger

log = get_logger("transcriber")


class Transcriber:
    """实时语音识别器"""

    def __init__(self):
        cfg = settings.get("transcriber")
        self.sample_rate = cfg.get("sample_rate", 16000)
        self.chunk_size = cfg.get("chunk_size", 1024)

        self.audio_processor = AudioProcessor(
            sample_rate=self.sample_rate,
            loudness_target=settings.get("audio.loudness_target", -20.0),
        )

        self.model = WhisperModel(
            cfg["model_path"],
            device=cfg.get("device", "cuda"),
            compute_type=cfg.get("compute_type", "float16"),
        )

        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.raw_audio = np.array([], dtype=np.float32)
        self.processing = False
        self.result_ready = threading.Event()
        self.current_transcript: Optional[str] = None
        self.stream: Optional[sd.InputStream] = None

        logger.info("语音识别模型已加载")

    def start_recording(self):
        """开始录音"""
        if self.is_recording or self.processing:
            return
        logger.info("开始录音...")
        self.is_recording = True
        self.raw_audio = np.array([], dtype=np.float32)
        self.result_ready.clear()

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size,
            callback=self._audio_callback,
            dtype=np.float32
        )
        self.stream.start()
        log.debug(f"录音流已启动: {self.sample_rate}Hz, {self.chunk_size} blocksize")

    def stop_recording(self):
        """停止录音并开始处理"""
        if not self.is_recording:
            return
        self.is_recording = False
        self.stream.stop()
        self.stream.close()

        queue_count = 0
        while not self.audio_queue.empty():
            self.raw_audio = np.append(self.raw_audio, self.audio_queue.get().flatten())
            queue_count += 1

        raw_duration = len(self.raw_audio) / self.sample_rate
        log.info(f"录音结束 | 原始音频: {raw_duration:.2f}s | 数据块: {queue_count}个")
        self.processing = True
        threading.Thread(target=self._processing_pipeline, daemon=True).start()

    def _audio_callback(self, indata, frames, time_info, status):
        """音频输入回调"""
        if status:
            log.warning(f"录音状态异常: {status}")
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def _processing_pipeline(self):
        """核心处理流程"""
        try:
            start_time = time.time()
            raw_duration = len(self.raw_audio) / self.sample_rate

            if raw_duration < settings.get("transcriber.min_audio_duration", 0.3):
                log.warning(f"录音时长过短 ({raw_duration:.2f}s)，已忽略")
                self.current_transcript = None
            else:
                log.debug(f"开始音频处理: {raw_duration:.2f}s → 降噪+响度标准化")
                processed_audio = self.audio_processor.process(self.raw_audio)
                log.debug(f"音频预处理完成")

                vad_cfg = settings.get("transcriber.vad", {})
                log.debug(f"开始语音识别 (beam_size={settings.get('transcriber.beam_size', 5)}, VAD开启)")
                segments, info = self.model.transcribe(
                    processed_audio,
                    language=settings.get("transcriber.language", "zh"),
                    beam_size=settings.get("transcriber.beam_size", 5),
                    vad_filter=True,
                    vad_parameters=VadOptions(
                        threshold=vad_cfg.get("threshold", 0.68),
                        min_speech_duration_ms=vad_cfg.get("min_speech_duration_ms", 400),
                        max_speech_duration_s=vad_cfg.get("max_speech_duration_s", 10),
                        min_silence_duration_ms=vad_cfg.get("min_silence_duration_ms", 700),
                        speech_pad_ms=vad_cfg.get("speech_pad_ms", 300),
                    )
                )
                self.current_transcript = " ".join([seg.text.strip() for seg in segments if seg.text])

                proc_time = time.time() - start_time
                duration_s = len(self.raw_audio) / self.sample_rate
                if self.current_transcript:
                    log.info(f"语音识别成功 | 「{self.current_transcript}」 | 耗时: {proc_time:.2f}s | 音频: {duration_s:.2f}s")
                else:
                    log.warning(f"语音识别完成但未检测到有效内容 | 耗时: {proc_time:.2f}s")
        except Exception as e:
            log.error(f"处理音频时发生错误: {e}")
            self.current_transcript = None
        finally:
            self.processing = False
            self.result_ready.set()

    def get_result(self) -> Optional[str]:
        """等待并获取识别结果"""
        self.result_ready.wait()
        result = self.current_transcript
        self.current_transcript = None
        self.result_ready.clear()
        return result
