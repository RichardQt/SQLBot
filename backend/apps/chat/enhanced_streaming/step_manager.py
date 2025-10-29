"""
步骤管理器
管理处理流程中的各个步骤
"""

import time
import uuid
from typing import Any

from .stream_events import (
    StepError,
    StepInfo,
    StepResult,
    StreamEvent,
    create_final_complete_event,
    create_step_complete_event,
    create_step_content_event,
    create_step_error_event,
    create_step_progress_event,
    create_step_start_event,
)


class StepManager:
    """步骤管理器"""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.sequence = 0
        self.current_step_id: str | None = None
        self.current_step_start_time: float | None = None
        self.steps_completed = 0
        self.steps_failed = 0
        self.total_start_time = time.time()

    def next_sequence(self) -> int:
        """获取下一个序列号"""
        self.sequence += 1
        return self.sequence

    def start_step(
        self,
        step_id: str,
        name: str,
        title: str,
        description: str,
        estimated_duration: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StreamEvent:
        """开始一个步骤"""
        self.current_step_id = step_id
        self.current_step_start_time = time.time()

        step_info = StepInfo(
            id=step_id,
            name=name,
            title=title,
            description=description,
            estimated_duration=estimated_duration,
            metadata=metadata or {},
        )

        return create_step_start_event(
            sequence=self.next_sequence(), session_id=self.session_id, step_info=step_info
        )

    def update_progress(
        self, step_id: str | None = None, progress: int = 0, message: str | None = None
    ) -> StreamEvent:
        """更新步骤进度"""
        step_id = step_id or self.current_step_id
        if not step_id:
            raise ValueError('No active step')

        return create_step_progress_event(
            sequence=self.next_sequence(),
            session_id=self.session_id,
            step_id=step_id,
            progress=progress,
            message=message,
        )

    def add_content(
        self, content: dict[str, Any], step_id: str | None = None
    ) -> StreamEvent:
        """添加步骤内容"""
        step_id = step_id or self.current_step_id
        if not step_id:
            raise ValueError('No active step')

        return create_step_content_event(
            sequence=self.next_sequence(),
            session_id=self.session_id,
            step_id=step_id,
            content=content,
        )

    def complete_step(
        self,
        summary: str,
        output: dict[str, Any] | None = None,
        step_id: str | None = None,
        next_step_info: tuple[str, str, str, str] | None = None,
    ) -> StreamEvent:
        """完成步骤"""
        step_id = step_id or self.current_step_id
        if not step_id:
            raise ValueError('No active step')

        # 计算耗时
        duration = 0
        if self.current_step_start_time:
            duration = int((time.time() - self.current_step_start_time) * 1000)

        result = StepResult(summary=summary, duration=duration, output=output)

        # 准备下一步信息
        next_step = None
        if next_step_info:
            next_step_id, next_name, next_title, next_desc = next_step_info
            next_step = StepInfo(
                id=next_step_id, name=next_name, title=next_title, description=next_desc
            )

        self.steps_completed += 1
        self.current_step_id = None
        self.current_step_start_time = None

        return create_step_complete_event(
            sequence=self.next_sequence(),
            session_id=self.session_id,
            step_id=step_id,
            result=result,
            next_step=next_step,
        )

    def fail_step(
        self,
        error_code: str,
        error_title: str,
        error_message: str,
        suggestion: str | None = None,
        can_retry: bool = False,
        step_id: str | None = None,
    ) -> StreamEvent:
        """步骤失败"""
        step_id = step_id or self.current_step_id
        if not step_id:
            raise ValueError('No active step')

        error = StepError(
            code=error_code,
            title=error_title,
            message=error_message,
            suggestion=suggestion,
            can_retry=can_retry,
        )

        self.steps_failed += 1
        self.current_step_id = None
        self.current_step_start_time = None

        return create_step_error_event(
            sequence=self.next_sequence(),
            session_id=self.session_id,
            step_id=step_id,
            error=error,
        )

    def finalize(self, output: dict[str, Any] | None = None) -> StreamEvent:
        """完成整个流程"""
        total_duration = int((time.time() - self.total_start_time) * 1000)

        summary = {
            'total_duration': total_duration,
            'completed_steps': self.steps_completed,
            'failed_steps': self.steps_failed,
            'output': output or {},
        }

        return create_final_complete_event(
            sequence=self.next_sequence(), session_id=self.session_id, summary=summary
        )
