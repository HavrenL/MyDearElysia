"""
AI 对话生成模块 — 调用 LLM API 生成爱莉希雅回复
"""
import json
import re
from collections import deque
from pathlib import Path
from typing import List, Dict

from openai import OpenAI

from src.config.settings import settings, ROOT_DIR
from src.utils.exceptions import DialogError
from src.utils.logger import logger, get_logger

log = get_logger("dialog")


class ContextManager:
    """对话上下文管理"""

    def __init__(self, max_rounds: int, system_prompt: str):
        self.max_messages = max_rounds * 2
        self.history: deque = deque(maxlen=self.max_messages + 1)
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def add_dialogue(self, user_input: str, ai_response: str):
        self.history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ai_response}
        ])

    def get_messages(self) -> List[Dict[str, str]]:
        return list(self.history)


class DialogGenerator:
    """AI 对话生成器"""

    def __init__(self):
        cfg = settings.get("dialog")

        # 加载角色 prompt
        character = cfg.get("character", "elysia")
        prompt_path = ROOT_DIR / "config" / "prompts" / f"{character}.json"
        if not prompt_path.exists():
            raise DialogError(f"角色配置不存在: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            character_cfg = json.load(f)

        system_prompt = character_cfg.get("system_prompt", "")

        # 追加说话风格
        style = character_cfg.get("speaking_style")
        if style:
            system_prompt += f"\n\n【说话风格】\n{style.strip()}"

        # 追加规则
        rules = character_cfg.get("rules", [])
        if rules:
            system_prompt += "\n\n" + "\n".join(f"- {r}" for r in rules)

        self.client = OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"]
        )
        self.model = cfg.get("model", "deepseek-ai/DeepSeek-V3")
        self.temperature = cfg.get("temperature", 1.0)
        self.max_tokens = cfg.get("max_tokens", 1024)

        self.ctx = ContextManager(
            max_rounds=cfg.get("max_rounds", 7),
            system_prompt=system_prompt
        )

        logger.info(f"对话模型已加载: {self.model} (角色: {character_cfg.get('name', character)})")

    @staticmethod
    def parse_emotion_tags(text: str):
        """解析回复中的情绪标签 {{开心}}，返回 [(情绪, 文本片段), ...]"""
        pattern = re.compile(r"\{\{(.+?)\}\}")
        parts = []
        last_end = 0
        current_emotion = None

        for m in pattern.finditer(text):
            # 标签前的纯文本
            if m.start() > last_end:
                plain = text[last_end:m.start()].strip()
                if plain:
                    parts.append((current_emotion, plain))
            current_emotion = m.group(1)
            last_end = m.end()

        # 尾部剩余文本
        tail = text[last_end:].strip()
        if tail:
            parts.append((current_emotion, tail))

        # 如果没有解析出任何标签，整段返回
        if not parts:
            parts = [(None, text.strip())]

        return parts

    def generate_response(self, user_input: str) -> str:
        """生成 AI 回复"""
        import time as time_module
        start_time = time_module.time()

        # 统计当前对话轮次
        history_count = len(self.ctx.get_messages()) // 2  # 排除 system
        log.info(f"向 AI 发送请求 | 模型: {self.model} | 历史: {history_count}轮 | 输入: 「{user_input[:80]}」")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.ctx.get_messages() + [{"role": "user", "content": user_input}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            ai_response = response.choices[0].message.content
            elapsed = time_module.time() - start_time

            # 尝试获取 token 用量
            usage = response.usage
            if usage:
                log.info(f"AI 回复完毕 | 耗时: {elapsed:.2f}s | "
                         f"输入: {usage.prompt_tokens}tok | 输出: {usage.completion_tokens}tok | "
                         f"总计: {usage.total_tokens}tok")
            else:
                log.info(f"AI 回复完毕 | 耗时: {elapsed:.2f}s")

            log.info(f"AI 回复内容: {ai_response}")
            self.ctx.add_dialogue(user_input, ai_response)
            return ai_response
        except Exception as e:
            elapsed = time_module.time() - start_time
            log.error(f"AI 请求失败 | 耗时: {elapsed:.2f}s | 错误: {e}")
            raise DialogError(f"生成回复失败: {e}")

    def get_context(self) -> List[Dict[str, str]]:
        return self.ctx.get_messages()
