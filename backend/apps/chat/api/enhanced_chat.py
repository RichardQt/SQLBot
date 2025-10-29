"""
增强的聊天API
提供详细步骤的流式响应
"""

from datetime import datetime

from common.core.deps import SessionDep, CurrentUser
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()


class EnhancedQuestionRequest(BaseModel):
    """增强问答请求"""

    question: str
    chat_id: int | None = None
    enhanced_streaming: bool = True


@router.post('/enhanced-question')
async def enhanced_question(
    request: EnhancedQuestionRequest,
    _session: SessionDep,
    _current_user: CurrentUser,
):
    """
    增强的流式问答接口
    返回详细的处理步骤
    """

    # 创建原始LLM服务(这里需要根据实际情况初始化)
    # llm_service = LLMService(...)

    # 创建增强的LLM服务
    # enhanced_service = EnhancedLLMService(llm_service, session)

    # 定义流式生成器
    async def generate_stream():
        """生成SSE流"""
        try:
            # 这是一个简化的演示实现
            # 实际应该使用 enhanced_service.stream_with_steps()

            # 示例: 发送步骤开始事件
            yield 'data: {"type": "step_start", "timestamp": "2025-10-28T10:00:00Z", "sequence": 1, "session_id": "test-session", "step": {"id": "step_001_analyzing", "name": "analyzing_question", "title": "分析问题意图", "description": "正在理解您的问题...", "estimated_duration": 2000}}\n\n'

            # 示例: 发送步骤进度事件
            yield 'data: {"type": "step_progress", "timestamp": "2025-10-28T10:00:01Z", "sequence": 2, "session_id": "test-session", "step_id": "step_001_analyzing", "progress": 50, "message": "正在分析问题语义..."}\n\n'

            # 示例: 发送步骤完成事件
            yield 'data: {"type": "step_complete", "timestamp": "2025-10-28T10:00:02Z", "sequence": 3, "session_id": "test-session", "step_id": "step_001_analyzing", "result": {"summary": "问题分析完成", "duration": 2000}, "next_step": {"id": "step_002_select_ds", "name": "select_datasource", "title": "选择数据源"}}\n\n'

            # 最终完成
            yield 'data: {"type": "final_complete", "timestamp": "2025-10-28T10:00:10Z", "sequence": 20, "session_id": "test-session", "summary": {"total_duration": 10000, "completed_steps": 6, "failed_steps": 0, "output": {"sql": "SELECT * FROM users", "explanation": "查询完成"}}}\n\n'

        except Exception as e:
            # 发送错误事件
            error_msg = f'{{"type": "stream_error", "timestamp": "{datetime.utcnow().isoformat()}Z", "sequence": 999, "session_id": "test-session", "error": {{"code": "INTERNAL_ERROR", "title": "处理失败", "message": "{str(e)}"}}}}\n\n'
            yield f'data: {error_msg}'

    return StreamingResponse(
        generate_stream(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )
