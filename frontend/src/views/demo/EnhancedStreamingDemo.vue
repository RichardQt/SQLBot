<template>
  <div class="enhanced-streaming-demo">
    <el-card class="demo-header">
      <h2>增强流式响应演示</h2>
      <p>此演示展示了详细的步骤进度展示功能</p>
    </el-card>

    <el-card class="demo-input">
      <el-input
        v-model="question"
        type="textarea"
        :rows="3"
        placeholder="请输入您的问题..."
        :disabled="isProcessing"
      />
      <div class="demo-actions">
        <el-button
          type="primary"
          :loading="isProcessing"
          :disabled="!question.trim()"
          @click="handleSubmit"
        >
          {{ isProcessing ? '处理中...' : '提交问题' }}
        </el-button>
        <el-button v-if="isProcessing" type="danger" @click="handleCancel">取消</el-button>
      </div>
    </el-card>

    <!-- 步骤进度展示 -->
    <StepProgress
      v-if="steps.length > 0"
      :steps="displaySteps"
      :overall-progress="overallProgress"
      :overall-title="overallTitle"
      :is-completed="isCompleted"
    />

    <!-- 最终结果 -->
    <el-card v-if="finalResult" class="demo-result">
      <h3>最终结果</h3>
      <pre>{{ JSON.stringify(finalResult, null, 2) }}</pre>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import StepProgress from '@/components/StepProgress.vue'
import { useEnhancedStreaming } from '@/composables/useEnhancedStreaming'

const question = ref('')
const finalResult = ref<any>(null)

const {
  steps,
  overallProgress,
  overallTitle,
  isCompleted,
  startEnhancedStreaming,
  cancelStreaming,
} = useEnhancedStreaming()

const isProcessing = computed(() => {
  return steps.length > 0 && !isCompleted.value
})

// 转换步骤数据格式以适配StepProgress组件
const displaySteps = computed(() => {
  return steps.map((step, index) => {
    // 映射状态
    let status: 'pending' | 'processing' | 'completed' | 'error' = 'pending'
    if (step.status === 'running') status = 'processing'
    else if (step.status === 'completed') status = 'completed'
    else if (step.status === 'error') status = 'error'

    return {
      id: index + 1,
      name: step.title,
      status,
      progress: step.progress,
      details: step.output,
      error: step.error?.message,
      result: step.metadata?.result ? JSON.stringify(step.metadata.result, null, 2) : undefined,
    }
  })
})

const handleSubmit = async () => {
  finalResult.value = null
  await startEnhancedStreaming(question.value)

  // 处理完成后提取最终结果
  if (isCompleted.value) {
    const lastStep = steps[steps.length - 1]
    if (lastStep?.metadata?.result) {
      finalResult.value = lastStep.metadata.result
    }
  }
}

const handleCancel = () => {
  cancelStreaming()
}
</script>

<style lang="less" scoped>
.enhanced-streaming-demo {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;

  .demo-header {
    margin-bottom: 20px;

    h2 {
      margin: 0 0 10px 0;
      color: var(--el-text-color-primary);
    }

    p {
      margin: 0;
      color: var(--el-text-color-secondary);
    }
  }

  .demo-input {
    margin-bottom: 20px;

    .demo-actions {
      margin-top: 12px;
      display: flex;
      gap: 12px;
    }
  }

  .demo-result {
    margin-top: 20px;

    h3 {
      margin: 0 0 12px 0;
    }

    pre {
      background: var(--el-fill-color-lighter);
      padding: 12px;
      border-radius: 4px;
      overflow-x: auto;
      margin: 0;
    }
  }
}
</style>
