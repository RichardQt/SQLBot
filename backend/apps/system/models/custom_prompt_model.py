"""自定义提示词模型。

注意：`custom_prompt` 表的 SQLModel 映射在 `sqlbot_xpack` 中已经定义。
本仓库只做轻量复用，避免重复定义同名表导致 SQLAlchemy 元数据冲突。
"""

from sqlbot_xpack.custom_prompt.models.custom_prompt_model import (  # noqa: F401
    CustomPrompt,
    CustomPromptInfo,
    CustomPromptTypeEnum,
)

__all__ = [
    "CustomPrompt",
    "CustomPromptInfo",
    "CustomPromptTypeEnum",
]
