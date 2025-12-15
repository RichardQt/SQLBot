/**
 * 文本清洗工具
 *
 * 目标：用于流式输出（reasoning_content 等）拼接与展示前的兜底，避免后端返回 null/undefined/None
 * 被 JS 隐式转换成字面字符串 "null" / "undefined" 并污染 UI。
 *
 * 注意：为了不破坏合法的 Markdown（例如代码块前导空格），这里不对有效文本做 trim 重写，
 * 只用 trim 进行“是否无意义”的判定。
 */

const NULL_LIKE_SET = new Set(['null', 'undefined', 'none'])

export const pickMeaningfulText = (value: unknown): string | undefined => {
  if (value === null || value === undefined) return undefined

  if (typeof value !== 'string') {
    const text = String(value)
    return pickMeaningfulText(text)
  }

  const trimmed = value.trim()
  if (!trimmed) return undefined

  const lowered = trimmed.toLowerCase()
  if (NULL_LIKE_SET.has(lowered)) return undefined

  return value
}

export const appendMeaningfulText = (base: string, chunk: unknown): string => {
  const text = pickMeaningfulText(chunk)
  if (!text) return base
  return base + text
}
