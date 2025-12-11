<script setup lang="ts">
import BaseAnswer from './BaseAnswer.vue'
import { Chat, chatApi, ChatInfo, type ChatMessage, ChatRecord, questionApi } from '@/api/chat.ts'
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import ChartBlock from '@/views/chat/chat-block/ChartBlock.vue'
import StepProgress from '@/components/StepProgress.vue'

// 定义步骤类型
interface ProcessingStep {
  id: number
  name: string
  status: 'pending' | 'processing' | 'completed' | 'error' | 'skipped'
  progress: number
  details?: string[]
  error?: string
  result?: string
  startTime?: number
  endTime?: number
  durationMs?: number
}

const props = withDefaults(
  defineProps<{
    chatList?: Array<ChatInfo>
    currentChatId?: number
    currentChat?: ChatInfo
    message?: ChatMessage
    loading?: boolean
    reasoningName: 'sql_answer' | 'chart_answer' | Array<'sql_answer' | 'chart_answer'>
  }>(),
  {
    chatList: () => [],
    currentChatId: undefined,
    currentChat: () => new ChatInfo(),
    message: undefined,
    loading: false,
  }
)

const emits = defineEmits([
  'finish',
  'error',
  'stop',
  'scrollBottom',
  'prefetchRecommend',
  'update:loading',
  'update:chatList',
  'update:currentChat',
  'update:currentChatId',
])

const index = computed(() => {
  if (props.message?.index) {
    return props.message.index
  }
  if (props.message?.index === 0) {
    return 0
  }
  return -1
})

const _currentChatId = computed({
  get() {
    return props.currentChatId
  },
  set(v) {
    emits('update:currentChatId', v)
  },
})

const _currentChat = computed({
  get() {
    return props.currentChat
  },
  set(v) {
    emits('update:currentChat', v)
  },
})

const _chatList = computed({
  get() {
    return props.chatList
  },
  set(v) {
    emits('update:chatList', v)
  },
})

const _loading = computed({
  get() {
    return props.loading
  },
  set(v) {
    emits('update:loading', v)
  },
})

const stopFlag = ref(false)

// 步骤进度状态
const steps = ref<ProcessingStep[]>([
  { id: 1, name: '分析问题', status: 'pending', progress: 0, details: [], result: '' },
  { id: 2, name: '选择数据源', status: 'pending', progress: 0, details: [], result: '' },
  { id: 3, name: '连接数据库', status: 'pending', progress: 0, details: [], result: '' },
  { id: 4, name: '生成SQL', status: 'pending', progress: 0, details: [], result: '' },
  { id: 5, name: '执行查询', status: 'pending', progress: 0, details: [], result: '' },
  { id: 6, name: '配置图表', status: 'pending', progress: 0, details: [], result: '' },
])
const overallProgress = ref(0)
const showSteps = ref(false)
const isCompleted = ref(false)

// 存储各步骤的实际输出内容
const stepResults = ref<Record<number, any>>({
  1: null, // 问题分析结果
  2: null, // 数据源信息
  3: null, // 数据库连接状态
  4: null, // 生成的SQL语句
  5: null, // 查询结果数据
  6: null, // 图表配置
})

const parseTimestamp = (value?: string): number | undefined => {
  if (!value) return undefined
  const timeValue = new Date(value).getTime()
  return Number.isNaN(timeValue) ? undefined : timeValue
}

const deriveDuration = (start?: number, end?: number, fallbackMs?: number): number | undefined => {
  if (typeof fallbackMs === 'number') {
    return fallbackMs
  }
  if (typeof start === 'number' && typeof end === 'number') {
    return Math.max(end - start, 0)
  }
  return undefined
}

// 处理步骤事件
const handleStepEvent = (data: any) => {
  showSteps.value = true

  const stepId = data.step_id
  const step = steps.value.find((s: ProcessingStep) => s.id === stepId)
  if (!step) return

  switch (data.type) {
    case 'step_start':
      step.status = 'processing'
      step.progress = 0
      step.startTime = parseTimestamp(data.started_at || data.timestamp)
      step.endTime = undefined
      step.durationMs = undefined
      step.error = undefined
      step.details = data.message ? [data.message] : []
      break
    case 'step_progress':
      step.progress = data.progress || 0
      if (!step.details) {
        step.details = []
      }
      if (data.message) {
        step.details.push(data.message)
      }
      break
    case 'step_content':
      if (!step.details) {
        step.details = []
      }
      if (data.content) {
        step.details.push(data.content)
      }
      break
    case 'step_complete':
      step.status = 'completed'
      step.progress = 100
      if (!step.details) {
        step.details = []
      }
      // 更新步骤名称为完成消息（如果包含详细信息如"问题分析完成（完整问题：xxxx）"）
      if (data.message) {
        step.name = data.message
        step.details.push(data.message)
      }
      step.endTime = parseTimestamp(data.finished_at || data.timestamp)
      if (!step.startTime) {
        step.startTime = parseTimestamp(data.started_at)
      }
      {
        const fallbackDuration =
          typeof data.duration_ms === 'number'
            ? data.duration_ms
            : typeof data.duration_seconds === 'number'
              ? Math.round(data.duration_seconds * 1000)
              : undefined
        const sanitizedFallback =
          typeof fallbackDuration === 'number' ? Math.max(fallbackDuration, 0) : undefined
        step.durationMs = deriveDuration(step.startTime, step.endTime, sanitizedFallback)
      }
      // 保存步骤结果数据
      if (data.result) {
        stepResults.value[stepId] = data.result
        // 特殊处理SQL执行结果（步骤5）
        if (stepId === 5) {
          const result = data.result
          let resultText = `查询返回 ${result.row_count} 行 ${result.col_count} 列`
          if (result.fields && result.fields.length > 0) {
            resultText += `\n\n列名: ${result.fields.join(', ')}`
          }
          if (result.preview_rows && result.preview_rows.length > 0) {
            resultText += `\n\n数据预览（前${result.preview_rows.length}行）:\n`
            resultText += JSON.stringify(result.preview_rows, null, 2)
          }
          step.result = resultText
        }
      }
      break
    case 'step_error':
      step.status = 'error'
      step.error = data.error || data.message
      step.endTime = parseTimestamp(data.finished_at || data.timestamp)
      if (!step.startTime) {
        step.startTime = parseTimestamp(data.started_at)
      }
      {
        const fallbackDuration =
          typeof data.duration_ms === 'number'
            ? data.duration_ms
            : typeof data.duration_seconds === 'number'
              ? Math.round(data.duration_seconds * 1000)
              : undefined
        const sanitizedFallback =
          typeof fallbackDuration === 'number' ? Math.max(fallbackDuration, 0) : undefined
        step.durationMs = deriveDuration(
          step.startTime,
          step.endTime ?? Date.now(),
          sanitizedFallback
        )
      }
      break
  }

  // 计算总体进度
  const completedSteps = steps.value.filter((s: ProcessingStep) => s.status === 'completed').length
  overallProgress.value = Math.floor((completedSteps / steps.value.length) * 100)

  // 检查是否全部完成（所有步骤都不处于 pending 或 processing 状态）
  const finishedSteps = steps.value.filter(
    (s: ProcessingStep) => s.status === 'completed' || s.status === 'error' || s.status === 'skipped'
  ).length
  isCompleted.value = finishedSteps === steps.value.length
}

// 跳过所有pending状态的步骤
const skipPendingSteps = (reason: string = '已跳过') => {
  steps.value.forEach((step: ProcessingStep) => {
    if (step.status === 'pending') {
      step.status = 'skipped'
      step.details = [reason]
      step.endTime = Date.now()
      step.durationMs = 0
    }
  })

  // 重新计算进度
  const completedSteps = steps.value.filter((s: ProcessingStep) => s.status === 'completed').length
  overallProgress.value = Math.floor((completedSteps / steps.value.length) * 100)

  // 检查是否全部完成（所有步骤都不处于 pending 或 processing 状态）
  const finishedSteps = steps.value.filter(
    (s: ProcessingStep) => s.status === 'completed' || s.status === 'error' || s.status === 'skipped'
  ).length
  isCompleted.value = finishedSteps === steps.value.length
}

// 标记正在处理中的步骤为失败,并跳过pending步骤
const markProcessingAsErrorAndSkipPending = (errorMessage: string) => {
  let hasProcessingStep = false

  steps.value.forEach((step: ProcessingStep) => {
    if (step.status === 'processing') {
      // 将正在进行的步骤标记为失败
      step.status = 'error'
      step.error = errorMessage
      step.endTime = Date.now()
      step.durationMs = deriveDuration(step.startTime, step.endTime)
      hasProcessingStep = true
    } else if (step.status === 'pending') {
      // 跳过待执行的步骤
      step.status = 'skipped'
      step.details = ['由于前序步骤失败,已自动跳过']
      step.endTime = Date.now()
      step.durationMs = 0
    }
  })

  // 重新计算进度
  const completedSteps = steps.value.filter((s: ProcessingStep) => s.status === 'completed').length
  overallProgress.value = Math.floor((completedSteps / steps.value.length) * 100)

  // 检查是否全部完成（所有步骤都不处于 pending 或 processing 状态）
  const finishedSteps = steps.value.filter(
    (s: ProcessingStep) => s.status === 'completed' || s.status === 'error' || s.status === 'skipped'
  ).length
  isCompleted.value = finishedSteps === steps.value.length

  return hasProcessingStep
}

const sendMessage = async () => {
  stopFlag.value = false
  _loading.value = true

  // 重置步骤状态
  steps.value.forEach((step: ProcessingStep) => {
    step.status = 'pending'
    step.progress = 0
    step.details = []
    step.error = undefined
    step.result = ''
    step.startTime = undefined
    step.endTime = undefined
    step.durationMs = undefined
  })
  overallProgress.value = 0
  isCompleted.value = false
  showSteps.value = true

  if (index.value < 0) {
    _loading.value = false
    showSteps.value = false
    return
  }

  const currentRecord: ChatRecord = _currentChat.value.records[index.value]

  let error: boolean = false
  if (_currentChatId.value === undefined) {
    error = true
  }
  if (error) return

  try {
    const controller: AbortController = new AbortController()
    const param = {
      question: currentRecord.question,
      chat_id: _currentChatId.value,
    }
    const response = await questionApi.add(param, controller)
    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')

    let sql_answer = ''
    let chart_answer = ''

    let tempResult = ''

    while (true) {
      if (stopFlag.value) {
        controller.abort()
        break
      }

      const { done, value } = await reader.read()
      if (done) {
        _loading.value = false
        break
      }

      let chunk = decoder.decode(value, { stream: true })
      tempResult += chunk
      const split = tempResult.match(/data:.*}\n\n/g)
      if (split) {
        chunk = split.join('')
        tempResult = tempResult.replace(chunk, '')
      } else {
        continue
      }
      if (chunk && chunk.startsWith('data:{')) {
        if (split) {
          for (const str of split) {
            let data
            try {
              data = JSON.parse(str.replace('data:{', '{'))
            } catch (err) {
              console.error('JSON string:', str)
              throw err
            }

            if (data.code && data.code !== 200) {
              ElMessage({
                message: data.msg,
                type: 'error',
                showClose: true,
              })
              _loading.value = false
              return
            }

            // 处理步骤事件
            if (
              [
                'step_start',
                'step_progress',
                'step_content',
                'step_complete',
                'step_error',
              ].includes(data.type)
            ) {
              handleStepEvent(data)
              continue
            }

            switch (data.type) {
              case 'id':
                currentRecord.id = data.id
                _currentChat.value.records[index.value].id = data.id
                break
              case 'info':
                console.info(data.msg)
                break
              case 'brief':
                _currentChat.value.brief = data.brief
                _chatList.value.forEach((c: Chat) => {
                  if (c.id === _currentChat.value.id) {
                    c.brief = _currentChat.value.brief
                  }
                })
                break
              case 'error':
                currentRecord.error = data.content
                // 当发生错误时,将正在进行的步骤标记为失败,并跳过所有pending的步骤
                markProcessingAsErrorAndSkipPending(data.content || '执行失败')
                // 如果是因为多轮对话未开启导致的问题不清晰，显示提示
                if (data.hint === 'enable_multi_turn') {
                  ElMessage({
                    message: '💡 开启多轮对话，可以更清晰的理解您的问题哦！',
                    type: 'info',
                    duration: 5000,
                    customClass: 'multi-turn-hint-message',
                  })
                }
                emits('error')
                break
              case 'datasource': {
                // 步骤2: 数据源选择结果
                stepResults.value[2] = {
                  id: data.id,
                  name: data.datasource_name,
                  engine_type: data.engine_type,
                }
                const step2 = steps.value.find((s: ProcessingStep) => s.id === 2)
                if (step2) {
                  step2.result = `数据源: ${data.datasource_name} (${data.engine_type})`
                }
                break
              }
              case 'sql-result':
                sql_answer += data.reasoning_content
                _currentChat.value.records[index.value].sql_answer = sql_answer
                break
              case 'sql': {
                // 步骤4: SQL生成结果
                _currentChat.value.records[index.value].sql = data.content
                stepResults.value[4] = data.content
                const step4 = steps.value.find((s: ProcessingStep) => s.id === 4)
                if (step4) {
                  step4.result = data.content
                }
                break
              }
              case 'sql-data':
                getChatData(_currentChat.value.records[index.value].id)
                // 提前触发预取推荐问题，不用等到整个流程完成
                emits('prefetchRecommend', currentRecord.id)
                break
              case 'chart-result':
                chart_answer += data.reasoning_content
                _currentChat.value.records[index.value].chart_answer = chart_answer
                break
              case 'chart': {
                // 步骤6: 图表配置结果
                _currentChat.value.records[index.value].chart = data.content
                stepResults.value[6] = data.content
                const step6 = steps.value.find((s: ProcessingStep) => s.id === 6)
                if (step6) {
                  const chartObj =
                    typeof data.content === 'string' ? JSON.parse(data.content) : data.content
                  step6.result = `图表类型: ${chartObj.type || '未知'}`
                }
                break
              }
              case 'finish':
                // Update log IDs for feedback feature
                if (data.sql_log_id) {
                  currentRecord.sql_log_id = data.sql_log_id
                }
                if (data.chart_log_id) {
                  currentRecord.chart_log_id = data.chart_log_id
                }
                // 当收到finish事件时,跳过所有pending的步骤
                skipPendingSteps('该问题不需要此步骤')
                emits('finish', currentRecord.id)
                break
            }
            await nextTick()
          }
        }
      }
    }
  } catch (error) {
    if (!currentRecord.error) {
      currentRecord.error = ''
    }
    if (currentRecord.error.trim().length !== 0) {
      currentRecord.error = currentRecord.error + '\n'
    }
    currentRecord.error = currentRecord.error + 'Error:' + error
    console.error('Error:', error)
    // 标记正在处理的步骤为失败,并跳过pending步骤
    markProcessingAsErrorAndSkipPending('请求处理异常')
    emits('error')
  } finally {
    _loading.value = false
  }
}

function getChatData(recordId?: number) {
  chatApi
    .get_chart_data(recordId)
    .then((response) => {
      _currentChat.value.records.forEach((record) => {
        if (record.id === recordId) {
          record.data = response
          // 步骤5: 查询结果数据
          stepResults.value[5] = response
          const step5 = steps.value.find((s: ProcessingStep) => s.id === 5)
          if (step5) {
            const rowCount = response?.data?.length || 0
            step5.result = `查询返回 ${rowCount} 行数据`
          }
        }
      })
    })
    .catch((error) => {
      // SQL执行失败时,标记步骤5为失败
      const step5 = steps.value.find((s: ProcessingStep) => s.id === 5)
      if (step5 && step5.status === 'processing') {
        step5.status = 'error'
        step5.error = error?.message || 'SQL执行失败'
        step5.endTime = Date.now()
        step5.durationMs = deriveDuration(step5.startTime, step5.endTime)
      }
      // 跳过后续pending步骤
      skipPendingSteps('由于SQL执行失败,后续步骤已自动跳过')
      console.error('获取查询数据失败:', error)
    })
    .finally(() => {
      emits('scrollBottom')
    })
}
function stop() {
  stopFlag.value = true
  _loading.value = false
  emits('stop')
}

onBeforeUnmount(() => {
  stop()
})

onMounted(() => {
  if (props.message?.record?.id && props.message?.record?.finish) {
    getChatData(props.message.record.id)
  }
})

defineExpose({ sendMessage, index: () => index.value, stop })
</script>

<template>
  <BaseAnswer v-if="message" :message="message" :reasoning-name="reasoningName" :loading="_loading">
    <!-- 步骤进度显示 -->
    <StepProgress
      v-if="showSteps"
      :steps="steps"
      :overall-progress="overallProgress"
      :is-completed="isCompleted"
      overall-title="SQL分析进度"
      style="margin-bottom: 16px"
    />

    <ChartBlock style="margin-top: 6px" :message="message" />
    <slot></slot>
    <template #tool>
      <slot name="tool"></slot>
    </template>
    <template #footer>
      <slot name="footer"></slot>
    </template>
  </BaseAnswer>
</template>

<style scoped lang="less"></style>

<style lang="less">
.multi-turn-hint-message {
  background-color: #f0f2f5 !important;
  border: 1px solid #e4e7ed !important;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1) !important;
  border-radius: 8px !important;
  min-width: 300px !important;

  .el-message__content {
    color: #606266 !important;
    font-weight: 500;
  }

  .el-message__icon {
    color: #909399 !important;
  }
}
</style>
