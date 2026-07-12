"""
TTS 合成模块 — 调用 GPT-SoVITS API 合成语音
"""
import io
import threading
from typing import Optional

import requests
import sounddevice as sd
import soundfile as sf

from src.audio.player import play_audio
from src.config.settings import settings
from src.utils.exceptions import TTSRequestError, TTSResponseError
from src.utils.logger import logger, get_logger

log = get_logger("tts")


class TTSClient:
    """GPT-SoVITS TTS 客户端"""

    def __init__(self):
        cfg = settings.get("tts")
        self.server_url = cfg["server_url"]
        self.default_params = {
            "text_lang": cfg.get("text_lang", "zh"),
            "ref_audio_path": cfg["ref_audio_path"],
            "prompt_lang": cfg.get("ref_audio_lang", "zh"),
            "prompt_text": cfg["ref_audio_text"],
            "top_k": cfg.get("top_k", 5),
            "top_p": cfg.get("top_p", 1),
            "temperature": cfg.get("temperature", 1),
            "text_split_method": cfg.get("text_split_method", "cut5"),
            "batch_size": cfg.get("batch_size", 1),
            "batch_threshold": cfg.get("batch_threshold", 0.75),
            "split_bucket": cfg.get("split_bucket", True),
            "speed_factor": cfg.get("speed_factor", 1.0),
            "fragment_interval": cfg.get("fragment_interval", 0.3),
            "seed": cfg.get("seed", -1),
            "media_type": cfg.get("media_type", "wav"),
            "streaming_mode": cfg.get("streaming_mode", False),
            "parallel_infer": cfg.get("parallel_infer", True),
            "repetition_penalty": cfg.get("repetition_penalty", 1.35),
        }
        self._timeout = cfg.get("request_timeout", 60)
        self._chunk_size = cfg.get("stream_chunk_size", 1024)
        log.info(f"TTS 客户端就绪 | 服务器: {self.server_url} | 超时: {self._timeout}s")
        log.debug(f"TTS 默认参数: {self.default_params}")

    def _set_weights(self, endpoint: str, weights_path: str) -> None:
        """切换模型权重"""
        url = f"{self.server_url}/{endpoint}?weights_path={weights_path}"
        log.info(f"正在切换权重 [{endpoint}]: {weights_path}")
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            err = resp.json()
            log.error(f"切换权重失败 [{endpoint}]: {err}")
            raise TTSResponseError(f"切换权重失败: {err}")
        log.info(f"权重切换成功 [{endpoint}]")

    def set_gpt_weights(self, weights_path: Optional[str] = None) -> None:
        """切换 GPT 模型权重"""
        path = weights_path or settings.get("tts.gpt_weights")
        log.info(f"准备切换 GPT 权重: {path}")
        self._set_weights("set_gpt_weights", path)
        log.info(f"GPT 权重已切换: {path.split('/')[-1]}")

    def set_sovits_weights(self, weights_path: Optional[str] = None) -> None:
        """切换 SoVITS 模型权重"""
        path = weights_path or settings.get("tts.sovits_weights")
        log.info(f"准备切换 SoVITS 权重: {path}")
        self._set_weights("set_sovits_weights", path)
        log.info(f"SoVITS 权重已切换: {path.split('/')[-1]}")

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """净化文本：移除 GPT-SoVITS 服务端无法编码的字符（如 ♪ ☆ ✿ 等装饰符号）

        GPT-SoVITS 的底层使用 GBK 编码，遇到 GBK 不支持的 Unicode 字符会报错。
        """
        # 已知会引发 GBK 编码错误的常见装饰符号
        forbidden_chars = set('♪♫♩♬♭♮♯☆★✿❀✦✧✩✪✫✬✭✮✯✰✨❄❅❆★☆🌈🌙⭐🌷🌸🌹🌺🌻🌼🌽🌾🌿🍀'
                              '💕💖💗💘💙💚💛💜💝💞💟❣❤🧡💋💌💎👑🎀🎁🎂🎃🎄🎅🎉🎊🎋🎌'
                              '🎍🎎🎏🎐🎑🎒🎓🎠🎡🎢🎣🎤🎥🎦🎧🎨🎩🎪🎫🎬🎭🎮🎯🎰🎱🎲🎳'
                              '🏆🏇🏈🏉🏊🏋🏌🏍🏎🏏🏐🏑🏒🏓🏔🏕🏖🏗🏘🏙🏚🏛🏜🏝🏞🏟🏠🏡🏢')
        sanitized = []
        for ch in text:
            if ch in forbidden_chars:
                continue
            try:
                ch.encode('gbk')
                sanitized.append(ch)
            except (UnicodeEncodeError, UnicodeDecodeError):
                # 尝试用空格替换无法编码的字符
                sanitized.append(' ')
        result = ''.join(sanitized)
        # 合并连续空格
        import re as _re
        result = _re.sub(r' +', ' ', result).strip()
        if result != text:
            log.debug(f"文本已净化: 删除了 {len(text) - len(result)} 个不支持的字符")
        return result

    def synthesize(self, text: str, **overrides) -> bytes:
        """合成语音并返回音频字节流"""
        import time as time_module
        start_time = time_module.time()

        # 最终安全网：清除所有花括号 + 净化不可编码字符
        text = text.replace('{', '').replace('}', '')
        text = self._sanitize_text(text)
        params = {**self.default_params, "text": text, **overrides}

        # 记录请求参数（debug 级别，避免日志过于冗长）
        ref_audio = params.get("ref_audio_path", "default")
        log.debug(f"TTS 请求: text=「{text[:60]}」 ref={ref_audio.split('/')[-1]} "
                  f"top_k={params.get('top_k')} temp={params.get('temperature')}")

        try:
            resp = requests.get(f"{self.server_url}/tts", params=params, stream=True, timeout=self._timeout)
        except requests.ConnectionError:
            log.error(f"无法连接到 TTS 服务器: {self.server_url}")
            raise TTSRequestError(f"无法连接到 TTS 服务器: {self.server_url}")

        if resp.status_code != 200:
            error = resp.json()
            log.error(f"TTS 合成失败 (HTTP {resp.status_code}): {error}")
            raise TTSResponseError(f"TTS 合成失败: {error}")

        audio_bytes = b""
        for chunk in resp.iter_content(chunk_size=self._chunk_size):
            if chunk:
                audio_bytes += chunk

        elapsed = time_module.time() - start_time
        duration = len(audio_bytes) / 1024
        log.info(f"TTS 合成成功 | {len(text)}字 → {duration:.1f}KB | 耗时: {elapsed:.2f}s")
        return audio_bytes

    def synthesize_fragment(self, text: str, ref_audio_path: str, ref_audio_text: str,
                            aux_ref_paths: list = None,
                            fragment_index: Optional[int] = None,
                            fragment_total: Optional[int] = None) -> bytes:
        """用指定的参考音频合成单个片段（支持多条辅助参考音频）"""
        ref_name = ref_audio_path.split('/')[-1] if '/' in ref_audio_path else ref_audio_path.split('\\')[-1]
        aux_info = f" +{len(aux_ref_paths)}条辅助" if aux_ref_paths else ""
        prefix = f" [{fragment_index}/{fragment_total}]" if fragment_index is not None and fragment_total is not None else ""
        log.info(f"合成片段{prefix}: 「{text[:40]}」 参考: {ref_name}{aux_info}")
        params = {
            "ref_audio_path": ref_audio_path,
            "prompt_text": ref_audio_text,
        }
        if aux_ref_paths:
            params["aux_ref_audio_paths"] = aux_ref_paths
        return self.synthesize(text, **params)

    def speak_fragments(self, fragments):
        """流水线合成播放：播第一段的同时合成第二段，消除等待间隙"""
        log.info(f"开始多片段流式播放, 共 {len(fragments)} 个片段")

        def _synth(frag, idx, total) -> bytes:
            text, ref_path, ref_text = frag[:3]
            aux_refs = frag[3] if len(frag) > 3 else None
            log.info(f"开始合成第 {idx}/{total} 段")
            audio = self.synthesize_fragment(text, ref_path, ref_text, aux_refs, idx, total)
            log.info(f"完成合成第 {idx}/{total} 段")
            return audio

        # 先合成第一段
        log.debug(f"预合成第 1/{len(fragments)} 段...")
        next_audio = _synth(fragments[0], 1, len(fragments))
        next_result = None
        next_thread = None

        for i in range(len(fragments)):
            audio = next_audio

            # 播当前片段的同时，在后台合成下一段
            if i + 1 < len(fragments):
                next_result = [None]
                def _bg(f, r, idx, total):
                    r[0] = _synth(f, idx, total)
                next_thread = threading.Thread(
                    target=_bg, args=(fragments[i + 1], next_result, i + 2, len(fragments)), daemon=True
                )
                next_thread.start()

            # 播放（播放期间下一段在后台合成）
            data, sr = sf.read(io.BytesIO(audio))
            log.info(f"开始播放第 {i + 1}/{len(fragments)} 段 ({len(data) / sr:.1f}s)")
            sd.play(data, sr)
            sd.wait()
            log.info(f"完成播放第 {i + 1}/{len(fragments)} 段")

            # 取下一段结果
            if next_thread and next_result:
                next_thread.join()
                next_audio = next_result[0]

        log.info("多片段流式播放完成")

    def play_from_first(self, first_audio: bytes, rest_fragments: list):
        """先播第一段，同时后台合成剩余片段，完成后无缝接上"""
        log.info(f"开始流水线播放: 1 段已合成 + {len(rest_fragments)} 段待合成")

        data, sr = sf.read(io.BytesIO(first_audio))
        log.debug(f"播放首段 ({len(data) / sr:.1f}s)，后台同步合成剩余 {len(rest_fragments)} 段")
        sd.play(data, sr)

        for idx, frag in enumerate(rest_fragments, start=2):
            result = [None]
            def _bg(f, r):
                text, ref_path, ref_text = f[:3]
                aux = f[3] if len(f) > 3 else None
                r[0] = self.synthesize_fragment(text, ref_path, ref_text, aux)
            t = threading.Thread(target=_bg, args=(frag, result), daemon=True)
            t.start()
            sd.wait()
            t.join()

            next_audio = result[0]
            if next_audio:
                d, s = sf.read(io.BytesIO(next_audio))
                log.debug(f"播放第 {idx} 段 ({len(d) / s:.1f}s)")
                sd.play(d, s)

        sd.wait()
        log.info("流水线播放完成")

    def speak(self, text: str, **overrides) -> None:
        """合成并播放语音"""
        log.info(f"开始单段合成播放: 「{text[:60]}」")
        try:
            audio = self.synthesize(text, **overrides)
            play_audio(audio)
            log.info("单段合成播放完成")
        except (TTSRequestError, TTSResponseError) as e:
            log.error(f"语音合成失败: {e}")
