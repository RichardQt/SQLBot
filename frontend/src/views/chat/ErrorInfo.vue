<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

const props = defineProps<{
  error?: string
}>()

const { t } = useI18n()

const showBlock = computed(() => {
  return props.error && props.error?.trim().length > 0
})

const errorType = computed(() => {
  if (showBlock.value && props.error?.trim().startsWith('{') && props.error?.trim().endsWith('}')) {
    try {
      const json = JSON.parse(props.error?.trim())
      return json['type']
    } catch (e) {
      console.error(e)
    }
  }
  return undefined
})

// 控制多轮对话提示框的显示
const showMultiTurnTip = ref(true)

// 关闭多轮对话提示框
const closeMultiTurnTip = () => {
  showMultiTurnTip.value = false
}

// 判断是否应该显示多轮对话提示框（生成SQL错误或执行SQL错误时显示）
const shouldShowMultiTurnTip = computed(() => {
  return (
    (errorType.value === 'generate-sql-err' || errorType.value === 'exec-sql-err') &&
    showMultiTurnTip.value
  )
})
</script>

<template>
  <div v-if="showBlock">
    <div class="error-container">
      <template v-if="errorType === 'db-connection-err'">
        {{ t('chat.ds_is_invalid') }}
      </template>
      <template v-else-if="errorType === 'exec-sql-err'">
        {{ t('chat.exec-sql-err-friendly') }}
      </template>
      <template v-else-if="errorType === 'generate-sql-err'">
        {{ t('chat.generate-sql-err-friendly') }}
      </template>
      <template v-else>
        {{ t('chat.error-friendly') }}
      </template>
    </div>

    <!-- 多轮对话引导提示框 - 在生成SQL错误或执行SQL错误时显示 -->
    <div v-if="shouldShowMultiTurnTip" class="multi-turn-tip">
      <div class="tip-content">
        <el-icon class="tip-icon"><InfoFilled /></el-icon>
        <span class="tip-text">{{ t('chat.multi_turn_tip') }}</span>
      </div>
      <el-icon class="close-icon" @click="closeMultiTurnTip"><Close /></el-icon>
    </div>
  </div>
</template>

<script lang="ts">
import { InfoFilled, Close } from '@element-plus/icons-vue'
export default {
  components: { InfoFilled, Close },
}
</script>

<style scoped lang="less">
.error-container {
  font-weight: 400;
  font-size: 16px;
  line-height: 24px;
  color: rgba(31, 35, 41, 1);
  white-space: pre-wrap;
  word-break: break-word;
}

.multi-turn-tip {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 12px;
  padding: 10px 14px;
  background: linear-gradient(135deg, #e8f4ff 0%, #f0f7ff 100%);
  border: 1px solid #b3d8ff;
  border-radius: 8px;

  .tip-content {
    display: flex;
    align-items: center;
    gap: 8px;

    .tip-icon {
      color: #409eff;
      font-size: 16px;
    }

    .tip-text {
      font-size: 14px;
      color: #606266;
      line-height: 20px;
    }
  }

  .close-icon {
    color: #909399;
    font-size: 14px;
    cursor: pointer;
    transition: color 0.2s;

    &:hover {
      color: #606266;
    }
  }
}
</style>
