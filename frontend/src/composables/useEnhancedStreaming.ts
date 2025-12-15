/**
 * 增强的流式响应处理
 * 提供详细的步骤进度展示
 */

import { ref, reactive, computed } from 'vue'
import type { Ref } from 'vue'
import { pickMeaningfulText } from '@/utils/text'

// 事件类型定义
type EventType =
  | 'step_start'
  | 'step_progress'
  | 'step_content'
  | 'step_complete'
  | 'step_error'
  | 'final_complete'
  | 'stream_error'

interface StepError {
  code: string
  title: string
  message: string
  suggestion?: string
  can_retry: boolean
}

interface StreamingStep {
  id: string
  name: string
  title: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'error'
  progress: number
  startTime?: string
  endTime?: string
  duration?: number
  output: string[]
  error?: StepError
  metadata: Record<string, any>
}

interface StreamingEvent {
  type: EventType
  timestamp: string
  sequence: number
  session_id: string
  step_id?: string
  step?: any
  content?: any
  result?: any
  error?: any
  next_step?: any
  summary?: any
  progress?: number
  message?: string
}

export function useEnhancedStreaming() {
  const steps = reactive<StreamingStep[]>([])
  const overallProgress = ref(0)
  const overallTitle = ref('准备开始')
  const isCompleted = ref(false)
  const abortController: Ref<AbortController | null> = ref(null)

  // 初始化步骤
  const initializeSteps = () => {
    steps.splice(0, steps.length)
    overallProgress.value = 0
    overallTitle.value = '准备开始'
    isCompleted.value = false
  }

  // 处理流式响应
  const processStream = async (response: Response) => {
    if (!response.body) {
      throw new Error('No response body')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || '' // 保留不完整的行

        for (const line of lines) {
          if (!line.startsWith('data:')) continue

          try {
            const data = JSON.parse(line.slice(5)) as StreamingEvent
            await handleStreamingEvent(data)
          } catch (error) {
            console.error('Failed to parse streaming data:', error, line)
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  }

  // 处理具体的流式事件
  const handleStreamingEvent = async (event: StreamingEvent) => {
    switch (event.type) {
      case 'step_start':
        if (event.step) {
          handleStepStart(event.step)
        }
        break

      case 'step_progress':
        if (event.step_id !== undefined && event.progress !== undefined) {
          updateStepProgress(event.step_id, event.progress, event.message)
        }
        break

      case 'step_content':
        if (event.step_id && event.content) {
          addStepOutput(event.step_id, event.content.text, event.content.reasoning)
        }
        break

      case 'step_complete':
        if (event.step_id) {
          completeStep(event.step_id, event.result)
          if (event.next_step) {
            prepareNextStep(event.next_step)
          }
        }
        updateOverallProgress()
        break

      case 'step_error':
        if (event.step_id && event.error) {
          handleStepError(event.step_id, event.error)
        }
        break

      case 'final_complete':
        isCompleted.value = true
        overallTitle.value = '处理完成'
        overallProgress.value = 100
        break

      case 'stream_error':
        handleGlobalError(event.error)
        break
    }
  }

  // 处理步骤开始
  const handleStepStart = (stepInfo: any) => {
    const existingStep = steps.find((s) => s.id === stepInfo.id)
    if (existingStep) {
      existingStep.status = 'running'
      existingStep.startTime = new Date().toISOString()
    } else {
      steps.push({
        id: stepInfo.id,
        name: stepInfo.name,
        title: stepInfo.title,
        description: stepInfo.description,
        status: 'running',
        progress: 0,
        startTime: new Date().toISOString(),
        output: [],
        metadata: stepInfo.metadata || {},
      })
    }
    overallTitle.value = stepInfo.title
  }

  // 更新步骤进度
  const updateStepProgress = (stepId: string, progress: number, message?: string) => {
    const step = steps.find((s) => s.id === stepId)
    if (step) {
      step.progress = progress
      if (message) {
        step.description = message
      }
    }
  }

  // 添加步骤输出
  const addStepOutput = (stepId: string, text: string, reasoning?: string) => {
    const step = steps.find((s) => s.id === stepId)
    if (step) {
      if (text) step.output.push(text)
      const reasoningText = pickMeaningfulText(reasoning)
      if (reasoningText) {
        // 处理思考过程的显示（过滤 null/undefined/none 等无意义内容）
        step.metadata.thinking = step.metadata.thinking || []
        step.metadata.thinking.push(reasoningText)
      }
    }
  }

  // 完成步骤
  const completeStep = (stepId: string, result?: any) => {
    const step = steps.find((s) => s.id === stepId)
    if (step) {
      step.status = 'completed'
      step.progress = 100
      step.endTime = new Date().toISOString()

      // 计算耗时
      if (step.startTime && step.endTime) {
        const start = new Date(step.startTime).getTime()
        const end = new Date(step.endTime).getTime()
        step.duration = end - start
      }

      if (result) {
        step.metadata.result = result
      }
    }
  }

  // 准备下一步
  const prepareNextStep = (nextStepInfo: any) => {
    const existingStep = steps.find((s) => s.id === nextStepInfo.id)
    if (existingStep) {
      // 已存在，等待开始
      return
    }

    // 添加新步骤
    steps.push({
      id: nextStepInfo.id,
      name: nextStepInfo.name,
      title: nextStepInfo.title,
      description: '等待执行...',
      status: 'pending',
      progress: 0,
      output: [],
      metadata: {},
    })
  }

  // 处理步骤错误
  const handleStepError = (stepId: string, errorInfo: any) => {
    const step = steps.find((s) => s.id === stepId)
    if (step) {
      step.status = 'error'
      step.error = {
        code: errorInfo.code,
        title: errorInfo.title,
        message: errorInfo.message,
        suggestion: errorInfo.suggestion,
        can_retry: errorInfo.can_retry,
      }
      step.endTime = new Date().toISOString()
    }
  }

  // 处理全局错误
  const handleGlobalError = (errorInfo: any) => {
    isCompleted.value = true
    overallTitle.value = '处理失败'
    console.error('Global error:', errorInfo)
  }

  // 更新总体进度
  const updateOverallProgress = () => {
    const completedSteps = steps.filter((s) => s.status === 'completed').length
    const totalSteps = steps.length
    if (totalSteps > 0) {
      overallProgress.value = Math.round((completedSteps / totalSteps) * 100)
    }

    if (completedSteps === totalSteps && totalSteps > 0) {
      isCompleted.value = true
      overallTitle.value = '处理完成'
    }
  }

  // 发起增强流式请求
  const startEnhancedStreaming = async (question: string, options?: { chatId?: number }) => {
    abortController.value = new AbortController()
    initializeSteps()
    overallTitle.value = '开始处理'

    try {
      const response = await fetch('/api/v1/chat/enhanced-question', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          chat_id: options?.chatId,
          enhanced_streaming: true,
        }),
        signal: abortController.value.signal,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      await processStream(response)
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.log('Streaming cancelled')
        overallTitle.value = '已取消'
      } else {
        handleGlobalError({
          code: 'NETWORK_ERROR',
          title: '网络错误',
          message: error.message,
        })
      }
    }
  }

  // 取消处理
  const cancelStreaming = () => {
    if (abortController.value) {
      abortController.value.abort()
    }
  }

  // 重试步骤
  const retryStep = async (stepId: string) => {
    const step = steps.find((s) => s.id === stepId)
    if (step && step.status === 'error') {
      step.status = 'pending'
      step.error = undefined
      step.output = []
      step.progress = 0

      // 这里可以发起特定步骤的重试请求
      // await retrySpecificStep(stepId)
    }
  }

  // 计算属性
  const currentStep = computed(() => {
    return steps.find((step) => step.status === 'running')
  })

  const hasFailedSteps = computed(() => {
    return steps.some((step) => step.status === 'error')
  })

  return {
    steps,
    overallProgress,
    overallTitle,
    isCompleted,
    currentStep,
    hasFailedSteps,
    startEnhancedStreaming,
    cancelStreaming,
    retryStep,
  }
}
