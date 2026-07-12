"""
参考音频管理器 — 按情绪标签匹配参考音频
"""
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config.settings import ROOT_DIR
from src.utils.logger import logger, get_logger

log = get_logger("reference")


def _get_emotion_aliases():
    """懒加载情绪别名映射（从配置读取）"""
    try:
        from src.config.settings import settings
        aliases = settings.get("reference.emotion_aliases")
        if aliases:
            return aliases
    except Exception:
        pass
    return {}


class ReferenceAudioManager:
    """参考音频管理器"""

    def __init__(self, audio_dir: str = None):
        if audio_dir is None:
            audio_dir = str(ROOT_DIR / "reference_audio")
        self.audio_dir = Path(audio_dir)
        self._cache: Dict[str, List[Tuple[str, str]]] = {}  # emotion -> [(path, text)]
        self._scan()

    def _scan(self):
        """扫描参考音频目录，按情绪分类"""
        if not self.audio_dir.exists():
            log.warning(f"参考音频目录不存在: {self.audio_dir}")
            return

        pattern = re.compile(r"【(.+?)】(.+)\.wav$")
        for f in sorted(self.audio_dir.iterdir()):
            if not f.name.endswith(".wav"):
                continue
            m = pattern.match(f.name)
            if m:
                emotion = m.group(1)
                text = m.group(2)
                if emotion not in self._cache:
                    self._cache[emotion] = []
                self._cache[emotion].append((str(f), text))

        # 统计情绪分布
        total = sum(len(v) for v in self._cache.values())
        distribution = ", ".join(f"{k}:{len(v)}条" for k, v in sorted(self._cache.items()))
        log.info(f"参考音频已加载: {total}条, {len(self._cache)}个情绪类别")
        log.debug(f"情绪分布: {distribution}")

    def get_emotions(self) -> List[str]:
        """获取所有可用的情绪标签"""
        return list(self._cache.keys())

    def _calc_score(self, text_a: str, text_b: str) -> float:
        """计算两条文本的内容相似度（基于关键词重叠）"""
        # 提取关键词：去掉常见语气词后按字符/词匹配
        skip = set("的了在是我有和就也都不到吧吗啊呀哦呢啦嘛~？！。，、")
        a_chars = set(c for c in text_a if c not in skip and not c.isspace())
        b_chars = set(c for c in text_b if c not in skip and not c.isspace())
        if not a_chars or not b_chars:
            return 0.0
        overlap = len(a_chars & b_chars)
        return overlap / max(len(a_chars), len(b_chars))

    def pick(self, emotion: str, target_text: str = "") -> Tuple[str, str]:
        """从情绪类别中选一条与目标文本最匹配的参考音频"""
        candidates = self._get_candidates(emotion)
        if candidates:
            result = self._pick_best(candidates, target_text)
            ref_name = result[0].split('\\')[-1].split('/')[-1]
            log.debug(f"选取参考音频 [{emotion}]: {ref_name}")
            return result

        from src.config.settings import settings
        default_path = settings.get("tts.ref_audio_path", "")
        default_text = settings.get("tts.ref_audio_text", "")
        log.warning(f"未找到情绪'{emotion}'的参考音频，使用默认音频")
        return default_path, default_text

    def pick_top_n(self, emotion: str, target_text: str = "", n: int = None) -> list:
        """取前 N 条最匹配的参考音频"""
        from src.config.settings import settings
        if n is None:
            n = settings.get("reference.top_n", 3)
        candidates = self._get_candidates(emotion)
        if not candidates:
            return [self.pick(emotion)]

        if target_text:
            scored = [(self._calc_score(target_text, ref[1]), ref) for ref in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [ref for _, ref in scored[:n]]
        return random.sample(candidates, min(n, len(candidates)))

    def _get_candidates(self, emotion: str) -> list:
        """获取情绪类别对应的参考音频候选列表"""
        if emotion in self._cache and self._cache[emotion]:
            return self._cache[emotion]
        # 通过别名查找
        aliases = _get_emotion_aliases()
        for std_emotion, alias_list in aliases.items():
            if emotion in alias_list and std_emotion in self._cache:
                return self._cache[std_emotion]
        return []

    def _pick_best(self, candidates: list, target_text: str = "") -> Tuple[str, str]:
        """从候选列表中选最匹配的一条"""
        if target_text:
            from src.config.settings import settings
            threshold = settings.get("reference.score_threshold", 0.05)
            scored = [(self._calc_score(target_text, ref[1]), ref) for ref in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0]
            if best[0] > threshold:
                return best[1]
        return random.choice(candidates)

    def pick_by_text(self, text: str) -> Tuple[str, str]:
        """根据文本内容智能匹配合适的参考音频"""
        # 优先【普通】情绪兜底，没有则用配置的 fallback
        for emotion in ["普通", "开心", "撒娇", "调皮", "感动"]:
            if emotion in self._cache and self._cache[emotion]:
                path, ref_text = random.choice(self._cache[emotion])
                return path, ref_text
        from src.config.settings import settings
        fallback = settings.get("reference.fallback_emotion", "开心")
        return self.pick(fallback)


# 全局单例
ref_audio = ReferenceAudioManager()
