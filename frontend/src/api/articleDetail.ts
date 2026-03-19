import { request } from '@/utils/request'

/**
 * 文章详情查询请求参数
 */
export interface ArticleQueryRequest {
  /** 点击数据的字段名 */
  field_name: string
  /** 点击数据的字段值 */
  field_value: string
  /**
   * 生成图表所用的原始 SQL 语句（可选）。
   * 当 field_name=unit_name 时，后端会自动从 WHERE 子句中提取
   * theme_name / Legal_topics 的过滤值作为主题日关键词。
   */
  original_sql?: string
  /** 数据源 ID（可选） */
  datasource_id?: number
  /** 分页参数 */
  page?: number
  page_size?: number
}

/**
 * 文章信息
 */
export interface ArticleInfo {
  article_title: string | null
  publish_time: string | null
  view_count: number
  article_url: string | null
  likes: number
  unit_name: string | null
  unit_property: string | null
}

/**
 * 文章查询响应
 */
export interface ArticleQueryResponse {
  success: boolean
  message: string
  data: ArticleInfo[]
  total: number
  page: number
  page_size: number
}

/**
 * 文章详情 API
 */
export const articleDetailApi = {
  /**
   * 查询文章详情
   */
  query: (params: ArticleQueryRequest): Promise<ArticleQueryResponse> =>
    request.post('/article_detail/query', params),
}
