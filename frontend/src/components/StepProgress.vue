<template>
  <el-card class="step-progress-card" shadow="never">
    <!-- 总体进度 -->
    <div class="overall-progress">
      <div class="progress-header">
        <span class="progress-title">{{ overallTitle }}</span>
        <el-tag :type="isCompleted ? 'success' : 'primary'" size="small">
          {{ isCompleted ? '已完成' : '进行中' }}
        </el-tag>
      </div>
      <el-progress
        :percentage="overallProgress"
        :status="isCompleted ? 'success' : undefined"
        :stroke-width="10"
      />
    </div>

    <!-- 详细步骤列表 -->
    <div class="steps-container">
      <el-collapse v-model="activeSteps" accordion>
        <el-collapse-item
          v-for="step in steps"
          :key="step.id"
          :name="step.id"
          :disabled="step.status === 'pending'"
        >
          <template #title>
            <div class="step-header">
              <!-- 步骤图标 -->
              <el-icon :class="['step-icon', `status-${step.status}`]" :size="20">
                <Loading v-if="step.status === 'processing'" class="rotating" />
                <Check v-else-if="step.status === 'completed'" />
                <Close v-else-if="step.status === 'error'" />
                <Clock v-else />
              </el-icon>

              <!-- 步骤标题 -->
              <span class="step-name">{{ step.name }}</span>

              <!-- 步骤状态标签 -->
              <el-tag :type="getStepTagType(step.status)" size="small" class="step-status-tag">
                {{ getStepStatusText(step.status) }}
              </el-tag>

              <!-- 步骤耗时 -->
              <span v-if="step.status !== 'pending'" class="step-duration">
                <template v-if="step.durationMs !== undefined">
                  耗时 {{ formatDuration(step.durationMs) }}
                </template>
                <template v-else-if="step.status === 'processing'"> 正在执行... </template>
                <template v-else> 耗时 -- </template>
              </span>
            </div>
          </template>

          <!-- 步骤详细内容 -->
          <div class="step-content">
            <!-- 步骤结果 -->
            <div v-if="step.result" class="step-result">
              <div class="result-label">
                <el-icon><DocumentCopy /></el-icon>
                <span>步骤结果:</span>
              </div>
              <div class="result-content">
                <!-- SQL代码高亮显示 (步骤4 - 生成SQL) -->
                <pre v-if="step.id === 4" class="sql-code"><code>{{ step.result }}</code></pre>
                <!-- SQL执行结果 (步骤5 - 执行查询) -->
                <pre
                  v-else-if="step.id === 5"
                  class="query-result"
                ><code>{{ step.result }}</code></pre>
                <!-- 其他文本结果 -->
                <div v-else class="text-result">{{ step.result }}</div>
              </div>
            </div>

            <!-- 步骤详细信息 -->
            <div v-if="step.details && step.details.length > 0" class="step-details">
              <div class="details-label">
                <el-icon><InfoFilled /></el-icon>
                <span>详细信息:</span>
              </div>
              <ul class="details-list">
                <li v-for="(detail, idx) in step.details" :key="idx">{{ detail }}</li>
              </ul>
            </div>

            <!-- 错误信息 -->
            <el-alert
              v-if="step.error"
              :title="step.error"
              type="error"
              :closable="false"
              show-icon
            />
          </div>
        </el-collapse-item>
      </el-collapse>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Loading, Check, Close, Clock, DocumentCopy, InfoFilled } from '@element-plus/icons-vue'

export interface Step {
  id: number
  name: string
  status: 'pending' | 'processing' | 'completed' | 'error'
  progress: number
  details?: string[]
  error?: string
  result?: string // 新增: 步骤的实际结果内容
  startTime?: number
  endTime?: number
  durationMs?: number
}

defineProps<{
  steps: Step[]
  overallProgress: number
  overallTitle: string
  isCompleted: boolean
}>()

const activeSteps = ref<number[]>([])

const getStepTagType = (status: string) => {
  switch (status) {
    case 'pending':
      return 'info'
    case 'processing':
      return 'primary'
    case 'completed':
      return 'success'
    case 'error':
      return 'danger'
    default:
      return 'info'
  }
}

const formatDuration = (durationMs?: number) => {
  if (typeof durationMs !== 'number') {
    return ''
  }
  if (durationMs < 1000) {
    return `${durationMs} ms`
  }
  const seconds = durationMs / 1000
  if (seconds < 60) {
    return `${seconds.toFixed(1)} s`
  }
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds - minutes * 60
  return `${minutes}m ${remainder.toFixed(1)}s`
}

const getStepStatusText = (status: string) => {
  switch (status) {
    case 'pending':
      return '等待中'
    case 'processing':
      return '进行中'
    case 'completed':
      return '已完成'
    case 'error':
      return '失败'
    default:
      return '未知'
  }
}
</script>

<style lang="less" scoped>
.step-progress-card {
  margin-bottom: 16px;
  border: 1px solid var(--el-border-color-lighter);

  .overall-progress {
    margin-bottom: 20px;

    .progress-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;

      .progress-title {
        font-size: 16px;
        font-weight: 500;
        color: var(--el-text-color-primary);
      }
    }
  }

  .steps-container {
    margin-top: 16px;

    .step-header {
      display: flex;
      align-items: center;
      gap: 12px;
      width: 100%;
      padding-right: 16px;

      .step-icon {
        flex-shrink: 0;

        &.status-pending {
          color: var(--el-color-info);
        }

        &.status-processing {
          color: var(--el-color-primary);
        }

        &.status-completed {
          color: var(--el-color-success);
        }

        &.status-error {
          color: var(--el-color-danger);
        }

        .rotating {
          animation: rotate 1s linear infinite;
          transform-origin: center;
        }
      }

      .step-name {
        flex: 1;
        font-size: 14px;
        font-weight: 500;
      }

      .step-status-tag {
        margin-left: auto;
      }

      .step-duration {
        margin-left: 12px;
        font-size: 12px;
        color: var(--el-text-color-secondary);
        white-space: nowrap;
      }
    }

    .step-content {
      padding: 16px 0 8px;

      .step-result {
        margin-bottom: 16px;
        border-left: 3px solid var(--el-color-primary);
        padding-left: 12px;

        .result-label {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 13px;
          font-weight: 500;
          color: var(--el-text-color-primary);
          margin-bottom: 8px;
        }

        .result-content {
          background: var(--el-fill-color-lighter);
          border-radius: 4px;
          padding: 12px;
          margin-top: 8px;

          .sql-code {
            margin: 0;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.6;
            color: var(--el-text-color-primary);
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 300px;
            overflow-y: auto;

            code {
              background: none;
              padding: 0;
            }
          }

          .query-result {
            margin: 0;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            line-height: 1.5;
            color: var(--el-text-color-primary);
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 400px;
            overflow-y: auto;
            background: var(--el-bg-color);
            padding: 8px;
            border-radius: 4px;

            code {
              background: none;
              padding: 0;
            }
          }

          .text-result {
            font-size: 13px;
            line-height: 1.6;
            color: var(--el-text-color-regular);
            word-break: break-word;
          }
        }
      }

      .step-details {
        margin-bottom: 12px;

        .details-label {
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 13px;
          font-weight: 500;
          color: var(--el-text-color-secondary);
          margin-bottom: 8px;
        }

        .details-list {
          list-style: none;
          padding: 0;
          margin: 0;

          li {
            padding: 4px 0;
            font-size: 13px;
            color: var(--el-text-color-regular);
          }
        }
      }
    }
  }
}

@keyframes rotate {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}
</style>
