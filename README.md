
MyDearElysia
===

**爱莉希雅 AI 语音交互助手** — 使用 GPT-SoVITS 语音合成 + DeepSeek/OpenAI API 驱动的角色扮演语音助手。


## 功能

- **🎤 语音对话** — 按住空格键说话，松开即识别并回复
- **⌨️ 文字聊天** — 直接输入文字与爱莉希雅对话
- **🎭 情绪语音** — 根据回复内容自动匹配合适的情绪参考音频，合成更有表现力的语音
- **🤖 AI 驱动** — 基于 DeepSeek API（兼容 OpenAI API），完全角色扮演

## 快速开始

### 1. 环境准备

- Python 3.10 ~ 3.11
- [FFmpeg](https://ffmpeg.org/download.html)（加入系统 PATH，或放在项目根目录 `ffmpeg/bin/` 下）
- NVIDIA GPU + CUDA（可选，但推荐）

### 2. 安装项目依赖

```bash
# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# 安装项目依赖
pip install -r requirements.txt

# 安装 GPT-SoVITS 依赖
pip install -r GPT_SoVITS/requirements.txt
```

### 3. 安装 PyTorch（GPU 版推荐）

```bash
# CUDA 12.x 用户
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# CPU 用户
pip install torch torchaudio
```

### 4. 下载模型

需要准备以下模型文件：

| 模型 | 路径 | 说明 |
|------|------|------|
| GPT-SoVITS 预训练模型 | `GPT_SoVITS/pretrained_models/` | 参考 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) |
| 微调权重（可选） | `weights/GPT/`、`weights/SoVITS/` | 你自己的微调模型 |
| faster-whisper 模型 | `whisper_models/faster-whisper-large-v3/` | 语音识别模型 |

### 5. 配置

复制配置模板并编辑：

```bash
cp config/app.example.json config/app.json
```

编辑 `config/app.json`，至少需要设置：

```json
{
  "dialog": {
    "api_key": "sk-your-api-key-here",   // 你的 API Key
    "base_url": "https://api.deepseek.com"  // API 地址
  },
  "tts": {
    "ref_audio_path": "reference_audio/你的参考音频.wav",
    "ref_audio_text": "参考音频的文字内容"
  }
}
```

参考音频建议准备多条不同情绪的语音（如开心、撒娇、难过等），放在 `reference_audio/` 下，文件名格式为 `【情绪】文本内容.wav`。

### 6. 运行

**步骤一：启动 TTS 服务**

```bash
python GPT_SoVITS/api_v2.py
```

> 首次启动需要加载模型，等待约 30~60 秒。如果使用自己的微调权重，可在 `GPT_SoVITS/configs/tts_infer.yaml` 中修改路径。

**步骤二：启动聊天程序**

新开一个终端：

```bash
python -m src.main
```

选择 **语音模式** 或 **打字模式** 即可开始对话。

**注意**：实际使用体验取决于硬件配置（GPU 性能、显存大小）以及所选 AI 模型的性能。低配硬件下语音合成和识别可能有明显延迟。

## 后续规划

- [ ] **🏠 微调本地模型** — 基于本地数据进行模型微调，逐步摆脱对云端 API 的依赖，实现完全离线运行
- [ ] **🖥️ 前端界面** — 开发图形化用户界面，提供更直观的交互体验，告别终端操作
- [ ] **🔧 高度自定义系统** — 支持灵活配置人物角色、语音参数、对话风格等，打造真正个性化的 AI 助手

## 项目结构

```
MyDearElysia/
├── src/                      # 核心代码
│   ├── main.py               # 入口，选择语音/文字模式
│   ├── core/
│   │   ├── dialog.py         # AI 对话引擎
│   │   ├── tts.py            # TTS 客户端（调用 GPT-SoVITS API）
│   │   └── transcriber.py    # 语音识别
│   ├── audio/
│   │   ├── reference.py      # 情绪参考音频管理
│   │   └── player.py         # 音频播放
│   ├── config/
│   │   └── settings.py       # 配置管理
│   └── utils/
│       ├── logger.py         # 日志
│       └── exceptions.py     # 异常定义
├── config/
│   ├── app.example.json      # 配置模板
│   └── prompts/
│       └── elysia.json       # 爱莉希雅角色提示词
├── GPT_SoVITS/               # GPT-SoVITS TTS 引擎
├── reference_audio/          # 参考音频（自行准备，参考 EXAMPLES.md）
├── weights/                  # 微调模型权重
├── whisper_models/           # faster-whisper 模型
├── requirements.txt          # 项目依赖
└── README.md
```

## 配置说明

完整配置项请参考 `config/app.example.json`，主要配置：

| 配置项 | 说明 |
|--------|------|
| `dialog.api_key` | DeepSeek / OpenAI API Key |
| `dialog.base_url` | API 地址 |
| `dialog.model` | 模型名称 |
| `dialog.character` | 角色，对应 `config/prompts/{character}.json` |
| `tts.server_url` | GPT-SoVITS 服务地址（默认 `http://127.0.0.1:9880`） |
| `tts.ref_audio_path` | 默认参考音频路径 |
| `tts.gpt_weights` | GPT 模型权重路径 |
| `tts.sovits_weights` | SoVITS 模型权重路径 |

## 参考音频命名规则

`reference_audio/` 下的音频文件按 `【情绪】文本内容.wav` 格式命名，系统会自动根据 AI 回复的情绪匹配合适的参考音频。支持的情绪标签：

开心、撒娇、调皮、感动、难过、撩拨、普通、惊喜、尴尬、生气、疲惫、积极、疑问、调侃的失望

## 技术栈

- **语音合成**：[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- **语音识别**：[faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- **AI 对话**：[DeepSeek API](https://platform.deepseek.com/) / OpenAI API
- **音频处理**：sounddevice, soundfile, librosa, noisereduce

## 许可

本项目基于 MIT 协议开源。

## 致谢

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) — 强大的语音合成引擎
- 《崩坏3》& 爱莉希雅
