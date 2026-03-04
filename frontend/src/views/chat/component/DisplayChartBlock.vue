<script setup lang="ts">
import ChartComponent from '@/views/chat/component/ChartComponent.vue'
import ArticleDetailDialog from '@/views/chat/component/ArticleDetailDialog.vue'
import type { ChatMessage } from '@/api/chat.ts'
import { computed, nextTick, ref, onMounted, onUnmounted } from 'vue'
import type { ChartTypes } from '@/views/chat/component/BaseChart.ts'
import { chartEventEmitter, type ChartClickEventData } from '@/views/chat/component/BaseG2Chart.ts'
import { useI18n } from 'vue-i18n'

const props = defineProps<{
  id?: number | string
  chartType: ChartTypes
  message: ChatMessage
  data: Array<{ [key: string]: any }>
}>()

const { t } = useI18n()

const chartObject = computed<{
  type: ChartTypes
  title: string
  axis: {
    x: { name: string; value: string }
    y: { name: string; value: string }
    series: { name: string; value: string }
  }
  columns: Array<{ name: string; value: string }>
}>(() => {
  if (props.message?.record?.chart) {
    return JSON.parse(props.message.record.chart)
  }
  return {}
})

const xAxis = computed(() => {
  if (chartObject.value?.axis?.x) {
    return [chartObject.value.axis.x]
  }
  return []
})
const yAxis = computed(() => {
  if (chartObject.value?.axis?.y) {
    return [chartObject.value.axis.y]
  }
  return []
})
const series = computed(() => {
  if (chartObject.value?.axis?.series) {
    return [chartObject.value.axis.series]
  }
  return []
})

const chartRef = ref()

// 文章详情弹窗状态
const articleDialogVisible = ref(false)
const articleFieldName = ref('')
const articleFieldValue = ref('')
const articleDialogTitle = ref('')

// 字段别名映射：将各种前端字段名映射到标准字段名
const FIELD_ALIAS_MAP: Record<string, string> = {
  // 行业系统相关
  industry_system_name: 'industry_system',
  industrySystem: 'industry_system',
  // 单位名称相关
  unitName: 'unit_name',
  // 单位属性相关
  unitProperty: 'unit_property',
  // 区域相关
  district: 'unit_district',
  area: 'unit_district',
  region: 'unit_district',
  // 法律法规相关
  legalContentType: 'legal_content_type',
  legal_type: 'legal_content_type',
  // 受众群体相关
  targetGroup: 'target_group',
  // 主题日相关
  themeName: 'theme_name',
  topic: 'theme_name',
}

// 支持的标准字段列表
const SUPPORTED_FIELDS = [
  'unit_name', // 单位名称
  'unit_property', // 单位属性
  'industry_system', // 行业系统
  'unit_district', // 单位区域(数据库字段)
  'district_name', // 区域名称(图表字段)
  'month', // 月份
  'legal_content_type', // 法律法规
  'target_group', // 受众群体
  'theme_name', // 主题日
  'Legal_topics', // 法律主题
]

/**
 * 规范化字段名：将别名转换为标准字段名
 */
function normalizeFieldName(fieldName: string): string {
  return FIELD_ALIAS_MAP[fieldName] || fieldName
}

/**
 * 检查字段是否受支持（包括别名）
 */
function isFieldSupported(fieldName: string): boolean {
  const normalized = normalizeFieldName(fieldName)
  return SUPPORTED_FIELDS.includes(normalized)
}

/**
 * 处理图表点击事件
 */
function handleChartClick(eventData: ChartClickEventData) {
  // 检查是否是当前图表的点击事件
  const chartId = `chart-component-${props.id ?? 'default_chat_id'}`
  if (eventData.chartId !== chartId) {
    return
  }

  console.log('DisplayChartBlock 收到图表点击事件:', eventData)

  // 尝试从点击数据中找到支持的字段
  let fieldName = ''
  let fieldValue = ''
  let displayName = ''

  // 优先检查 x 轴字段（确保 value 有效，排除 undefined/null）
  if (eventData.x && eventData.x.value != null && isFieldSupported(eventData.x.field)) {
    fieldName = normalizeFieldName(eventData.x.field)
    fieldValue = String(eventData.x.value)
    displayName = eventData.x.name
  }
  // 然后检查系列字段
  else if (
    eventData.series &&
    eventData.series.value != null &&
    isFieldSupported(eventData.series.field)
  ) {
    fieldName = normalizeFieldName(eventData.series.field)
    fieldValue = String(eventData.series.value)
    displayName = eventData.series.name
  }
  // 最后检查原始数据中的字段
  else if (eventData.rawData) {
    // 对于折线图，rawData 可能是数组，需要取第一个元素
    let dataToCheck = eventData.rawData
    if (
      eventData.chartType === 'line' &&
      Array.isArray(eventData.rawData) &&
      eventData.rawData.length > 0
    ) {
      dataToCheck = eventData.rawData[0]
    }
    // 遍历 rawData 中的所有字段，检查是否有支持的字段（包括别名）
    for (const rawField of Object.keys(dataToCheck)) {
      if (dataToCheck[rawField] && isFieldSupported(rawField)) {
        fieldName = normalizeFieldName(rawField)
        fieldValue = String(dataToCheck[rawField])
        displayName = rawField
        break
      }
    }
  }

  // 如果找到了支持的字段，显示文章详情弹窗
  if (fieldName && fieldValue) {
    articleFieldName.value = fieldName
    articleFieldValue.value = fieldValue
    articleDialogTitle.value = `${displayName}: ${fieldValue} - 文章详情`
    articleDialogVisible.value = true
  } else {
    console.log('未找到支持的字段，不显示文章详情弹窗')
  }
}

// 当前消息对应的原始 SQL（用于下钻时提取主题日等条件）
const originalSql = computed<string>(() => {
  return (props.message?.record as any)?.sql ?? ''
})

// 注册事件监听
onMounted(() => {
  chartEventEmitter.on('chart-click', handleChartClick as any)
})

onUnmounted(() => {
  chartEventEmitter.off('chart-click', handleChartClick as any)
})

function onTypeChange() {
  nextTick(() => {
    chartRef.value?.destroyChart()
    chartRef.value?.renderChart()
  })
}
function getViewInfo() {
  return {
    chart: {
      columns: chartObject.value?.columns,
      type: props.chartType,
      xAxis: xAxis.value,
      yAxis: yAxis.value,
      series: series.value,
      title: chartObject.value.title,
    },
    data: { data: props.data },
  }
}
function getExcelData() {
  return chartRef.value?.getExcelData()
}

defineExpose({
  onTypeChange,
  getViewInfo,
  getExcelData,
})
</script>

<template>
  <div v-if="message.record?.chart" class="chart-base-container">
    <ChartComponent
      v-if="message.record.id && data?.length > 0"
      :id="id ?? 'default_chat_id'"
      ref="chartRef"
      :type="chartType"
      :columns="chartObject?.columns"
      :x="xAxis"
      :y="yAxis"
      :series="series"
      :data="data"
    />
    <el-empty v-else :description="t('chat.no_data')" />

    <!-- 文章详情弹窗 -->
    <ArticleDetailDialog
      v-model:visible="articleDialogVisible"
      :field-name="articleFieldName"
      :field-value="articleFieldValue"
      :title="articleDialogTitle"
      :original-sql="originalSql"
    />
  </div>
</template>

<style scoped lang="less">
.chart-base-container {
  height: 100%;
  width: 100%;
  border-radius: 12px;
  background: rgba(224, 224, 226, 0.29);
}
</style>
