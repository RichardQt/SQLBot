<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { articleDetailApi, type ArticleInfo, type ArticleQueryRequest } from '@/api/articleDetail'
import { ElMessage } from 'element-plus'

const props = defineProps<{
  visible: boolean
  fieldName: string
  fieldValue: string
  title?: string
  /** 生成图表的原始 SQL，后端从中提取主题日等 WHERE 条件 */
  originalSql?: string
}>()

const emit = defineEmits<{
  (e: 'update:visible', value: boolean): void
}>()

// 响应式数据
const loading = ref(false)
const articleList = ref<ArticleInfo[]>([])
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(10)

// 计算属性：对话框标题
const dialogTitle = computed(() => {
  return props.title || `${props.fieldValue} - 文章详情`
})

// 计算属性：对话框可见性
const dialogVisible = computed({
  get: () => props.visible,
  set: (value) => emit('update:visible', value),
})

// 查询文章数据
async function fetchArticles() {
  if (!props.fieldName || !props.fieldValue) {
    return
  }

  loading.value = true
  try {
    const params: ArticleQueryRequest = {
      field_name: props.fieldName,
      field_value: props.fieldValue,
      original_sql: props.originalSql || undefined,
      page: currentPage.value,
      page_size: pageSize.value,
    }

    const response = await articleDetailApi.query(params)
    if (response.success) {
      articleList.value = response.data
      total.value = response.total
    } else {
      ElMessage.warning(response.message || '查询失败')
      articleList.value = []
      total.value = 0
    }
  } catch (error: any) {
    console.error('查询文章详情失败:', error)
    ElMessage.error('查询文章详情失败')
    articleList.value = []
    total.value = 0
  } finally {
    loading.value = false
  }
}

// 分页变化
function handleCurrentChange(page: number) {
  currentPage.value = page
  fetchArticles()
}

function handleSizeChange(size: number) {
  pageSize.value = size
  currentPage.value = 1
  fetchArticles()
}

// 打开文章链接
function openArticleUrl(url: string | null) {
  if (url) {
    window.open(url, '_blank')
  }
}

// 格式化日期
function formatDate(dateStr: string | null): string {
  if (!dateStr) return '-'
  try {
    const date = new Date(dateStr)
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return dateStr
  }
}

// 格式化数字
function formatNumber(num: number | null | undefined): string {
  if (num === null || num === undefined) return '0'
  return num.toLocaleString()
}

// 监听对话框打开
watch(
  () => props.visible,
  (newVal) => {
    if (newVal) {
      currentPage.value = 1
      fetchArticles()
    }
  }
)
</script>

<template>
  <el-dialog
    v-model="dialogVisible"
    :title="dialogTitle"
    width="900px"
    destroy-on-close
    :close-on-click-modal="false"
  >
    <div v-loading="loading" class="article-list-container">
      <!-- 文章列表 -->
      <el-table :data="articleList" stripe style="width: 100%" max-height="400">
        <el-table-column
          prop="article_title"
          label="文章标题"
          min-width="280"
          show-overflow-tooltip
        >
          <template #default="{ row }">
            <span
              v-if="row.article_url"
              class="article-link"
              @click="openArticleUrl(row.article_url)"
            >
              {{ row.article_title || '-' }}
            </span>
            <span v-else>{{ row.article_title || '-' }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="publish_time" label="发布时间" width="160">
          <template #default="{ row }">
            {{ formatDate(row.publish_time) }}
          </template>
        </el-table-column>
        <el-table-column prop="view_count" label="阅读量" width="100" align="center">
          <template #default="{ row }">
            {{ formatNumber(row.view_count) }}
          </template>
        </el-table-column>
        <el-table-column prop="likes" label="点赞数" width="100" align="center">
          <template #default="{ row }">
            {{ formatNumber(row.likes) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="80" align="center">
          <template #default="{ row }">
            <el-button
              v-if="row.article_url"
              type="primary"
              link
              size="small"
              @click="openArticleUrl(row.article_url)"
            >
              查看
            </el-button>
            <span v-else class="no-link">-</span>
          </template>
        </el-table-column>
      </el-table>

      <!-- 空状态 -->
      <el-empty v-if="!loading && articleList.length === 0" description="暂无文章数据" />

      <!-- 分页 -->
      <div v-if="total > 0" class="pagination-container">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :pager-count="7"
          :page-sizes="[10, 20, 50, 100]"
          :total="total"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </div>

    <template #footer>
      <div class="dialog-footer">
        <el-button @click="dialogVisible = false">关闭</el-button>
      </div>
    </template>
  </el-dialog>
</template>

<style scoped lang="less">
.article-list-container {
  min-height: 200px;
}

.article-link {
  color: var(--el-color-primary);
  cursor: pointer;
  &:hover {
    text-decoration: underline;
  }
}

.no-link {
  color: var(--el-text-color-placeholder);
}

.pagination-container {
  margin-top: 16px;
  display: flex;
  justify-content: flex-end;
}

.dialog-footer {
  text-align: right;
}
</style>
