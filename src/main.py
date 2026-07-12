"""
MyDearElysia — 爱莉希雅 AI 语音交互助手

启动方式：
    1. 先启动 TTS 服务:  python GPT_SoVITS/api_v2.py ...
    2. 再运行本程序:    python -m src.main
"""
import os
import re
import sys

# 确保项目根目录在 sys.path 中
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import threading

from pynput import keyboard

from src.audio.reference import ref_audio
from src.config.settings import settings
from src.core.dialog import DialogGenerator
from src.core.tts import TTSClient
from src.core.transcriber import Transcriber
from src.utils.logger import logger


def build_fragments(reply: str, tts: TTSClient):
    """将 AI 回复解析为带情绪参考音频的合成片段
    
    Returns:
        list of (text, ref_audio_path, ref_audio_text, [aux_ref_paths])
    """
    parts = DialogGenerator.parse_emotion_tags(reply)
    fragments = []
    emotion_seq = []

    for emotion, text in parts:
        if not text:
            continue

        # 去掉可能残留的花括号标签
        text = text.replace('{', '').replace('}', '').strip()
        if not text:
            continue

        if emotion:
            emotion_seq.append(emotion)
            # 取 top 3：第一条做主参考，其余做辅助参考
            top_refs = ref_audio.pick_top_n(emotion, target_text=text, n=3)
            ref_path, ref_text = top_refs[0]
            aux_paths = [p for p, _ in top_refs[1:]] if len(top_refs) > 1 else None
        else:
            emotion_seq.append("默认")
            # 无标签时用默认
            ref_path = settings.get("tts.ref_audio_path")
            ref_text = settings.get("tts.ref_audio_text")
            aux_paths = None

        fragments.append((text, ref_path, ref_text, aux_paths))

    # 情绪序列日志
    if len(emotion_seq) > 1:
        seq_str = " → ".join(emotion_seq)
        logger.info(f"情绪序列: [{seq_str}] ({len(fragments)}个片段)")

    return fragments


def speak_with_emotion(reply: str, tts: TTSClient):
    """带情绪地合成并播放回复"""
    fragments = build_fragments(reply, tts)
    logger.info(f"开始播放: {len(fragments)}个片段")

    if len(fragments) <= 1:
        # 单片段直接播（用清理后的文本，不用原始 reply）
        text, ref_path, ref_text, aux_refs = fragments[0]
        tts.speak(text, ref_audio_path=ref_path, prompt_text=ref_text)
    else:
        # 多片段流式播放
        tts.speak_fragments(fragments)


def choose_mode() -> str:
    """让用户选择输入模式"""
    print("\n" + "=" * 40)
    print("  MyDearElysia — 爱莉希雅 AI 助手")
    print("=" * 40)
    print("  1. 🎤  语音模式（按住空格说话）")
    print("  2. ⌨️  打字模式（直接输入文字）")
    print("=" * 40)
    while True:
        choice = input("  请选择 [1/2]: ").strip()
        if choice == "1":
            logger.info("用户选择: 语音模式")
            print("  → 已选择语音模式\n")
            return "voice"
        elif choice == "2":
            logger.info("用户选择: 打字模式")
            print("  → 已选择打字模式\n")
            return "text"
        print("  输入无效，请重新选择")


def run_voice_mode(dialog: DialogGenerator, tts: TTSClient):
    """语音输入模式"""
    transcriber = Transcriber()

    logger.info("系统初始化完成！按住空格键说话")

    def on_press(key):
        if key == keyboard.Key.space:
            transcriber.start_recording()

    def on_release(key):
        if key == keyboard.Key.space:
            transcriber.stop_recording()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    while True:
        transcript = transcriber.get_result()
        if not transcript:
            continue

        logger.info(f"你说: {transcript}")
        try:
            reply = dialog.generate_response(transcript)
            if reply:
                speak_with_emotion(reply, tts)
        except Exception as e:
            logger.error(f"处理出错: {e}")


def run_text_mode(dialog: DialogGenerator, tts: TTSClient):
    """文字输入模式"""
    logger.info("系统初始化完成！输入文字和爱莉希雅聊天吧（输入 exit 退出）")
    print("\n💬 ", end="", flush=True)

    while True:
        try:
            text = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见~")
            break

        if not text:
            continue
        if text.lower() in ("exit", "quit", "退出"):
            print("再见~ 下次也要来找我玩哦 💕")
            break

        logger.info(f"你说: {text}")
        try:
            reply = dialog.generate_response(text)
            if reply:
                print(f"\n💕 {reply}")
                speak_with_emotion(reply, tts)
                print("\n💬 ", end="", flush=True)
        except Exception as e:
            logger.error(f"处理出错: {e}")
            print("\n💬 ", end="", flush=True)


def main():
    import time as time_module
    start_time = time_module.time()

    logger.info("=" * 50)
    logger.info("MyDearElysia 启动中...")

    # 加载配置
    settings.load()

    # 按配置更新日志
    from src.utils.logger import configure_from_settings
    configure_from_settings()

    # 初始化公共模块
    logger.info("初始化 AI 对话引擎...")
    dialog = DialogGenerator()

    logger.info("初始化 TTS 客户端...")
    tts = TTSClient()

    logger.info("切换 GPT 权重...")
    tts.set_gpt_weights()
    logger.info("切换 SoVITS 权重...")
    tts.set_sovits_weights()

    elapsed = time_module.time() - start_time
    logger.info(f"系统初始化完成 | 耗时: {elapsed:.2f}s")

    # 选择模式
    mode = choose_mode()
    if mode == "voice":
        run_voice_mode(dialog, tts)
    else:
        run_text_mode(dialog, tts)


if __name__ == "__main__":
    main()
