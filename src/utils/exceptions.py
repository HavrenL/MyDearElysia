"""
自定义异常
"""


class TranscriberError(Exception):
    """语音识别相关错误"""
    pass


class DialogError(Exception):
    """AI 对话相关错误"""
    pass


class TTSRequestError(Exception):
    """TTS 请求相关错误"""
    pass


class TTSResponseError(Exception):
    """TTS 响应异常"""
    pass


class ConfigError(Exception):
    """配置加载错误"""
    pass
