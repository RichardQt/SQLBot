"""
流式事件定义
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StreamEventType(str, Enum):
    """流式事件类型"""

    STEP_START = 'step_start'
    STEP_PROGRESS = 'step_progress'
    STEP_CONTENT = 'step_content'
    STEP_COMPLETE = 'step_complete'
    STEP_ERROR = 'step_error'
    FINAL_COMPLETE = 'final_complete'
    STREAM_ERROR = 'stream_error'


class StepInfo(BaseModel):
    """步骤信息"""

    id: str
    name: str
    title: str
    description: str
    estimated_duration: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepProgress(BaseModel):
    """步骤进度"""

    progress: int  # 0-100
    message: str | None = None


class StepResult(BaseModel):
    """步骤结果"""

    summary: str
    duration: int
    output: dict[str, Any] | None = None


class StepError(BaseModel):
    """步骤错误"""

    code: str
    title: str
    message: str
    suggestion: str | None = None
    can_retry: bool = False


class StreamEvent(BaseModel):
    """流式事件基类"""

    type: StreamEventType
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    sequence: int
    session_id: str
    step_id: str | None = None
    step: StepInfo | None = None
    progress: int | None = None
    message: str | None = None
    content: dict[str, Any] | None = None
    result: StepResult | None = None
    error: StepError | None = None
    next_step: StepInfo | None = None
    summary: dict[str, Any] | None = None

    def to_sse(self) -> str:
        """转换为SSE格式"""
        return f'data: {self.model_dump_json()}\n\n'


def create_step_start_event(
    sequence: int, session_id: str, step_info: StepInfo
) -> StreamEvent:
    """创建步骤开始事件"""
    return StreamEvent(
        type=StreamEventType.STEP_START,
        sequence=sequence,
        session_id=session_id,
        step=step_info,
    )


def create_step_progress_event(
    sequence: int, session_id: str, step_id: str, progress: int, message: str | None = None
) -> StreamEvent:
    """创建步骤进度事件"""
    return StreamEvent(
        type=StreamEventType.STEP_PROGRESS,
        sequence=sequence,
        session_id=session_id,
        step_id=step_id,
        progress=progress,
        message=message,
    )


def create_step_content_event(
    sequence: int, session_id: str, step_id: str, content: dict[str, Any]
) -> StreamEvent:
    """创建步骤内容事件"""
    return StreamEvent(
        type=StreamEventType.STEP_CONTENT,
        sequence=sequence,
        session_id=session_id,
        step_id=step_id,
        content=content,
    )


def create_step_complete_event(
    sequence: int,
    session_id: str,
    step_id: str,
    result: StepResult,
    next_step: StepInfo | None = None,
) -> StreamEvent:
    """创建步骤完成事件"""
    return StreamEvent(
        type=StreamEventType.STEP_COMPLETE,
        sequence=sequence,
        session_id=session_id,
        step_id=step_id,
        result=result,
        next_step=next_step,
    )


def create_step_error_event(
    sequence: int, session_id: str, step_id: str, error: StepError
) -> StreamEvent:
    """创建步骤错误事件"""
    return StreamEvent(
        type=StreamEventType.STEP_ERROR,
        sequence=sequence,
        session_id=session_id,
        step_id=step_id,
        error=error,
    )


def create_final_complete_event(
    sequence: int, session_id: str, summary: dict[str, Any]
) -> StreamEvent:
    """创建最终完成事件"""
    return StreamEvent(
        type=StreamEventType.FINAL_COMPLETE,
        sequence=sequence,
        session_id=session_id,
        summary=summary,
    )


def create_stream_error_event(
    sequence: int, session_id: str, error: StepError
) -> StreamEvent:
    """创建流错误事件"""
    return StreamEvent(
        type=StreamEventType.STREAM_ERROR,
        sequence=sequence,
        session_id=session_id,
        error=error,
    )
