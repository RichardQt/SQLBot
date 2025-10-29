# 增强流式响应 - 步骤展示功能

## 概述

增强流式响应功能提供了详细的处理步骤展示,让用户能够实时了解系统处理问题的每个阶段。

## 功能特性

### 前端组件

1. **StepProgress.vue** - 步骤进度展示组件

   - 总体进度条显示
   - 详细步骤列表
   - 每个步骤的状态(等待中/进行中/已完成/失败)
   - 步骤结果和详细信息展示
   - 支持折叠/展开步骤详情

2. **useEnhancedStreaming.ts** - 增强流式处理 Composable
   - 管理流式事件处理
   - 步骤状态追踪
   - 进度计算
   - 错误处理

### 后端模块

1. **enhanced_streaming/** - 增强流式响应模块

   - `stream_events.py` - 流式事件定义
   - `step_manager.py` - 步骤管理器
   - `enhanced_llm_service.py` - 增强的 LLM 服务(待实现)

2. **enhanced_chat.py** - 增强聊天 API
   - `/api/v1/chat/enhanced-question` - 增强问答接口
   - 返回 SSE(Server-Sent Events)格式的流式响应

## 事件类型

系统支持以下流式事件类型:

- `step_start` - 步骤开始
- `step_progress` - 步骤进度更新
- `step_content` - 步骤内容输出
- `step_complete` - 步骤完成
- `step_error` - 步骤错误
- `final_complete` - 全部完成
- `stream_error` - 流错误

## 使用示例

### 前端使用

```vue
<template>
	<StepProgress
		:steps="displaySteps"
		:overall-progress="overallProgress"
		:overall-title="overallTitle"
		:is-completed="isCompleted"
	/>
</template>

<script setup>
import { useEnhancedStreaming } from "@/composables/useEnhancedStreaming";
import StepProgress from "@/components/StepProgress.vue";

const {
	steps,
	overallProgress,
	overallTitle,
	isCompleted,
	startEnhancedStreaming,
	cancelStreaming,
} = useEnhancedStreaming();

// 开始处理
await startEnhancedStreaming("查询用户数据");

// 取消处理
cancelStreaming();
</script>
```

### 后端使用

```python
from apps.chat.enhanced_streaming import StepManager

# 创建步骤管理器
step_manager = StepManager()

# 开始步骤
event = step_manager.start_step(
    step_id="step_001",
    name="analyzing",
    title="分析问题",
    description="正在分析问题意图..."
)
yield event.to_sse()

# 更新进度
event = step_manager.update_progress(progress=50, message="分析中...")
yield event.to_sse()

# 完成步骤
event = step_manager.complete_step(summary="分析完成")
yield event.to_sse()

# 最终完成
event = step_manager.finalize(output={"result": "success"})
yield event.to_sse()
```

## 演示页面

访问 `/demo/enhanced-streaming` 查看功能演示。

## 待完成事项

- [ ] 实现完整的 EnhancedLLMService
- [ ] 集成到实际的聊天流程中
- [ ] 添加步骤重试功能
- [ ] 优化错误处理和用户提示
- [ ] 添加更多步骤类型和状态

## 技术栈

- 前端: Vue 3 + TypeScript + Element Plus
- 后端: FastAPI + Pydantic
- 通信: Server-Sent Events (SSE)
