"""
图表点击事件 - 文章详情查询 API
根据图表点击数据查询相关文章的详细信息
"""
import re
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

logger = logging.getLogger(__name__)

from common.core.deps import SessionDep, CurrentUser
from apps.datasource.crud.datasource import get_datasource_list
from apps.db.db import exec_sql

router = APIRouter(tags=["article_detail"], prefix="/article_detail")


class ArticleQueryRequest(BaseModel):
    """文章查询请求参数"""
    # 点击数据的字段名
    field_name: str
    # 点击数据的字段值
    field_value: str
    # 原始 SQL 语句（可选），后端会自动从中提取主题日关键词
    # 当 field_name=unit_name 且 SQL 的 WHERE 子句包含主题日过滤条件时生效
    original_sql: Optional[str] = None
    # 数据源 ID（可选，默认使用第一个数据源）
    datasource_id: Optional[int] = None
    # 分页参数
    page: int = 1
    page_size: int = 10


class ArticleInfo(BaseModel):
    """文章信息"""
    article_title: Optional[str] = None
    publish_time: Optional[str] = None
    view_count: Optional[int] = None
    article_url: Optional[str] = None
    likes: Optional[int] = None
    unit_name: Optional[str] = None
    unit_property: Optional[str] = None


class ArticleQueryResponse(BaseModel):
    """文章查询响应"""
    success: bool
    message: str
    data: List[ArticleInfo] = []
    total: int = 0
    page: int = 1
    page_size: int = 10


# 字段名别名映射：将前端各种字段名统一映射到标准字段名
FIELD_ALIAS_MAPPING = {
    # 行业系统相关别名
    'industry_system_name': 'industry_system',
    'industrySystem': 'industry_system',
    'industry': 'industry_system',
    # 单位名称相关别名
    'unitName': 'unit_name',
    'name': 'unit_name',
    # 单位属性相关别名
    'unitProperty': 'unit_property',
    'property': 'unit_property',
    # 区域相关别名
    'district': 'unit_district',
    'area': 'unit_district',
    'region': 'unit_district',
    # 法律法规相关别名
    'legalContentType': 'legal_content_type',
    'legal_type': 'legal_content_type',
    # 受众群体相关别名
    'targetGroup': 'target_group',
    'group': 'target_group',
    # 主题日相关别名
    'themeName': 'theme_name',
    'topic': 'theme_name',
}

# 字段到表的映射关系
FIELD_TABLE_MAPPING = {
    # 单位名称和单位属性 -> fx_education_articles
    'unit_name': 'fx_education_articles',
    'unit_property': 'fx_education_articles',
    # 行业系统 -> fx_education_articles
    'industry_system': 'fx_education_articles',
    # 单位区域 -> fx_education_articles
    'unit_district': 'fx_education_articles',
    # 区域名称（图表字段）-> fx_education_articles（映射到 unit_district）
    'district_name': 'fx_education_articles',
    # 月份 -> fx_education_articles
    'month': 'fx_education_articles',
    # 法律法规 -> fx_education_articles_legal
    'legal_content_type': 'fx_education_articles_legal',
    # 受众群体 -> fx_education_articles_group
    'target_group': 'fx_education_articles_group',
    # 主题日 -> fx_education_articles_legal (Legal_topics)
    'theme_name': 'fx_education_articles_legal',
    'Legal_topics': 'fx_education_articles_legal',
}


# 下钻基础查询中始终存在的表别名（无需额外 JOIN）
_DRILL_BASE_ALIASES = {'r', 'e'}

# 额外表的标准 JOIN 语句（按依赖顺序定义，t 依赖 l）
_EXTRA_JOIN_MAP = {
    'l': 'JOIN fx_education_articles_legal l ON l.article_id = e.article_id',
    't': "JOIN fx_theme t ON l.Legal_topics LIKE CONCAT('%', t.theme_name, '%')",
    'g': 'JOIN fx_education_articles_group g ON g.article_id = e.article_id',
}

# 下钻时需要跳过的 WHERE 条件（基础查询中已包含）
_SKIP_CONDITION_PATTERNS = [
    re.compile(r'`?(?:\w+`?\.`?)?type_class`?\s*=', re.IGNORECASE),
]


def _split_where_conditions(sql: str) -> List[str]:
    """
    提取原始 SQL 的 WHERE 子句，按顶层 AND 分割为独立条件列表。
    通过括号深度跟踪，避免将函数参数或子查询内部的 AND 误判为分隔符。
    """
    where_match = re.search(
        r'\bWHERE\b(.*?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|$)',
        sql, re.IGNORECASE | re.DOTALL
    )
    if not where_match:
        return []

    where_clause = where_match.group(1).strip()
    conditions: List[str] = []
    depth = 0
    current = ''
    i = 0

    while i < len(where_clause):
        ch = where_clause[i]
        if ch == '(':
            depth += 1
            current += ch
            i += 1
        elif ch == ')':
            depth -= 1
            current += ch
            i += 1
        elif depth == 0 and where_clause[i:i + 3].upper() == 'AND':
            # 确保 AND 前后不是字母/数字/下划线，避免误匹配标识符
            before_ok = i == 0 or not (where_clause[i - 1].isalnum() or where_clause[i - 1] == '_')
            after_ok = (i + 3 >= len(where_clause)) or not (where_clause[i + 3].isalnum() or where_clause[i + 3] == '_')
            if before_ok and after_ok:
                cond = current.strip()
                if cond:
                    conditions.append(cond)
                current = ''
                i += 3
                continue
            else:
                current += ch
                i += 1
        else:
            current += ch
            i += 1

    cond = current.strip()
    if cond:
        conditions.append(cond)

    return conditions


def extract_drill_conditions(sql: str, normalized_field: str) -> dict:
    """
    通用下钻条件提取：从原始聚合 SQL 的 WHERE 子句中自动提取下钻所需的
    额外 JOIN 和 WHERE 条件，无需针对每种筛选类型单独编码。

    工作原理：
      1. 分割原始 SQL 的顶层 WHERE 条件
      2. 跳过基础查询已有的条件（如 type_class、被聚合字段本身）
      3. 检测剩余条件引用的表别名，按依赖顺序推断所需额外 JOIN

    适用场景示例：
      - WHERE DATE_FORMAT(r.publish_time, '%Y-%m') = '2025-06'   → 月份过滤
      - WHERE t.theme_name LIKE '%国家公祭日%'                    → 主题日过滤
      - 任何其他基于原始 SQL WHERE 条件的下钻需求

    返回:
        {
            'extra_joins':      ['JOIN fx_theme t ON ...', ...],
            'extra_conditions': ['t.theme_name LIKE ...', 'DATE_FORMAT(...) = ...']
        }
    """
    result: dict = {'extra_joins': [], 'extra_conditions': []}
    if not sql:
        return result

    all_conditions = _split_where_conditions(sql)

    # 跳过被聚合字段本身的等值条件（防御性处理）
    skip_field_pattern = re.compile(
        r'`?(?:\w+`?\.`?)?' + re.escape(normalized_field) + r'`?\s*=',
        re.IGNORECASE
    )

    extra_conditions: List[str] = []
    for cond in all_conditions:
        cond = cond.strip()
        if not cond:
            continue
        if any(p.search(cond) for p in _SKIP_CONDITION_PATTERNS):
            continue
        if skip_field_pattern.search(cond):
            continue
        extra_conditions.append(cond)

    if not extra_conditions:
        return result

    # 检测额外条件中引用的所有表别名，推断所需额外 JOIN
    all_text = ' '.join(extra_conditions)
    referenced = set(re.findall(r'`?(\w+)`?\s*\.\s*`?\w+`?', all_text))
    extra_aliases = referenced - _DRILL_BASE_ALIASES

    extra_joins: List[str] = []
    # l 与 t 需按依赖顺序添加（t 的 JOIN ON 条件依赖 l）
    if 'l' in extra_aliases or 't' in extra_aliases:
        extra_joins.append(_EXTRA_JOIN_MAP['l'])
    if 't' in extra_aliases:
        extra_joins.append(_EXTRA_JOIN_MAP['t'])
    if 'g' in extra_aliases:
        extra_joins.append(_EXTRA_JOIN_MAP['g'])

    result['extra_joins'] = extra_joins
    result['extra_conditions'] = extra_conditions
    return result


def normalize_field_name(field_name: str) -> str:
    """将前端字段名规范化为标准字段名"""
    # 先查别名映射
    if field_name in FIELD_ALIAS_MAPPING:
        return FIELD_ALIAS_MAPPING[field_name]
    # 如果没有别名，返回原字段名
    return field_name


def get_table_by_field(field_name: str) -> Optional[str]:
    """根据字段名获取对应的表名"""
    # 先规范化字段名
    normalized_field = normalize_field_name(field_name)
    return FIELD_TABLE_MAPPING.get(normalized_field)


def build_query_sql(table_name: str, field_name: str, field_value: str, offset: int, limit: int,
                    extra_joins: Optional[List[str]] = None,
                    extra_conditions: Optional[List[str]] = None) -> Optional[str]:
    """
    根据表名和字段构建文章详情查询 SQL。
    extra_joins / extra_conditions 由 extract_drill_conditions() 自动提供，
    无需针对月份、主题日等场景分别编码。
    """
    field_name = normalize_field_name(field_name)
    extra_joins = extra_joins or []
    extra_conditions = extra_conditions or []

    # theme_name 字段走固定关联 SQL（使用模糊匹配）
    if field_name == 'theme_name':
        sql = f"""
            SELECT DISTINCT
                r.article_title,
                r.publish_time,
                r.view_count,
                r.article_url,
                r.thumbs_count AS likes,
                e.unit_name,
                e.unit_property
            FROM fx_theme t
            JOIN fx_education_articles_legal l ON l.Legal_topics LIKE CONCAT('%', t.theme_name, '%')
            JOIN fx_education_articles e ON l.article_id = e.article_id
            JOIN fx_article_records r ON e.article_id = r.article_id
            WHERE t.status = 1
              AND e.type_class = '1'
              AND t.theme_name LIKE '%{field_value}%'
              AND r.publish_time >= t.start_date
              AND r.publish_time < DATE_ADD(t.end_date, INTERVAL 1 DAY)
            ORDER BY r.publish_time DESC
            LIMIT {limit} OFFSET {offset}
        """
        return sql

    if field_name == 'district_name':
        field_name = 'unit_district'

    # 动态拼接额外 JOIN 和 WHERE 片段
    join_fragment = ('\n            ' + '\n            '.join(extra_joins)) if extra_joins else ''
    cond_fragment = ('\n              AND ' + '\n              AND '.join(extra_conditions)) if extra_conditions else ''

    if table_name == 'fx_education_articles':
        sql = f"""
            SELECT DISTINCT
                r.article_title,
                r.publish_time,
                r.view_count, 
                r.article_url,
                r.thumbs_count AS likes,
                e.unit_name,
                e.unit_property
            FROM fx_education_articles e
            JOIN fx_article_records r ON e.article_id = r.article_id{join_fragment}
            WHERE e.type_class = '1'
              AND e.{field_name} = '{field_value}'{cond_fragment}
            ORDER BY r.publish_time DESC
            LIMIT {limit} OFFSET {offset}
        """
    elif table_name == 'fx_education_articles_legal':
        # l 已在 FROM 中，过滤重复 JOIN
        _legal_joins = [j for j in extra_joins if 'fx_education_articles_legal' not in j]
        _join_frag = ('\n            ' + '\n            '.join(_legal_joins)) if _legal_joins else ''
        _cond_frag = ('\n              AND ' + '\n              AND '.join(extra_conditions)) if extra_conditions else ''
        sql = f"""
            SELECT DISTINCT
                r.article_title,
                r.publish_time,
                r.view_count,
                r.article_url,
                r.thumbs_count AS likes,
                e.unit_name,
                e.unit_property
            FROM fx_education_articles_legal l
            JOIN fx_education_articles e ON l.article_id = e.article_id
            JOIN fx_article_records r ON e.article_id = r.article_id{_join_frag}
            WHERE e.type_class = '1'
              AND l.{field_name} = '{field_value}'{_cond_frag}
            ORDER BY r.publish_time DESC
            LIMIT {limit} OFFSET {offset}
        """
    elif table_name == 'fx_education_articles_group':
        # g 已在 FROM 中，过滤重复 JOIN
        _group_joins = [j for j in extra_joins if 'fx_education_articles_group' not in j]
        _join_frag = ('\n            ' + '\n            '.join(_group_joins)) if _group_joins else ''
        _cond_frag = ('\n              AND ' + '\n              AND '.join(extra_conditions)) if extra_conditions else ''
        sql = f"""
            SELECT DISTINCT
                r.article_title,
                r.publish_time,
                r.view_count,
                r.article_url,
                r.thumbs_count AS likes,
                e.unit_name,
                e.unit_property
            FROM fx_education_articles_group g
            JOIN fx_education_articles e ON g.article_id = e.article_id
            JOIN fx_article_records r ON e.article_id = r.article_id{_join_frag}
            WHERE e.type_class = '1'
              AND g.{field_name} = '{field_value}'{_cond_frag}
            ORDER BY r.publish_time DESC
            LIMIT {limit} OFFSET {offset}
        """
    else:
        return None

    return sql


def build_count_sql(table_name: str, field_name: str, field_value: str,
                    extra_joins: Optional[List[str]] = None,
                    extra_conditions: Optional[List[str]] = None) -> Optional[str]:
    """构建计数 SQL，extra_joins / extra_conditions 与 build_query_sql 保持一致。"""
    field_name = normalize_field_name(field_name)
    extra_joins = extra_joins or []
    extra_conditions = extra_conditions or []

    # theme_name 字段走固定关联 SQL（使用模糊匹配）
    if field_name == 'theme_name':
        sql = f"""
            SELECT COUNT(DISTINCT r.article_id) AS total
            FROM fx_theme t
            JOIN fx_education_articles_legal l ON l.Legal_topics LIKE CONCAT('%', t.theme_name, '%')
            JOIN fx_education_articles e ON l.article_id = e.article_id
            JOIN fx_article_records r ON e.article_id = r.article_id
            WHERE t.status = 1
              AND e.type_class = '1'
              AND t.theme_name LIKE '%{field_value}%'
              AND r.publish_time >= t.start_date
              AND r.publish_time < DATE_ADD(t.end_date, INTERVAL 1 DAY)
        """
        return sql

    if field_name == 'district_name':
        field_name = 'unit_district'

    join_fragment = ('\n            ' + '\n            '.join(extra_joins)) if extra_joins else ''
    cond_fragment = ('\n              AND ' + '\n              AND '.join(extra_conditions)) if extra_conditions else ''

    if table_name == 'fx_education_articles':
        sql = f"""
            SELECT COUNT(DISTINCT r.article_id) AS total
            FROM fx_education_articles e
            JOIN fx_article_records r ON e.article_id = r.article_id{join_fragment}
            WHERE e.type_class = '1'
              AND e.{field_name} = '{field_value}'{cond_fragment}
        """
    elif table_name == 'fx_education_articles_legal':
        _legal_joins = [j for j in extra_joins if 'fx_education_articles_legal' not in j]
        _join_frag = ('\n            ' + '\n            '.join(_legal_joins)) if _legal_joins else ''
        _cond_frag = ('\n              AND ' + '\n              AND '.join(extra_conditions)) if extra_conditions else ''
        sql = f"""
            SELECT COUNT(DISTINCT r.article_id) AS total
            FROM fx_education_articles_legal l
            JOIN fx_education_articles e ON l.article_id = e.article_id
            JOIN fx_article_records r ON e.article_id = r.article_id{_join_frag}
            WHERE e.type_class = '1'
              AND l.{field_name} = '{field_value}'{_cond_frag}
        """
    elif table_name == 'fx_education_articles_group':
        _group_joins = [j for j in extra_joins if 'fx_education_articles_group' not in j]
        _join_frag = ('\n            ' + '\n            '.join(_group_joins)) if _group_joins else ''
        _cond_frag = ('\n              AND ' + '\n              AND '.join(extra_conditions)) if extra_conditions else ''
        sql = f"""
            SELECT COUNT(DISTINCT r.article_id) AS total
            FROM fx_education_articles_group g
            JOIN fx_education_articles e ON g.article_id = e.article_id
            JOIN fx_article_records r ON e.article_id = r.article_id{_join_frag}
            WHERE e.type_class = '1'
              AND g.{field_name} = '{field_value}'{_cond_frag}
        """
    else:
        return None

    return sql


@router.post("/query", response_model=ArticleQueryResponse)
async def query_article_detail(
    session: SessionDep,
    current_user: CurrentUser,
    request: ArticleQueryRequest
):
    """
    根据图表点击数据查询文章详情
    
    1. 根据字段名判断属于哪张表
    2. 关联 fx_education_articles 和 fx_article_records 获取文章详情
    3. 返回：article_title, publish_time, view_count, article_url, likes
    """
    try:
        # 获取字段对应的表名
        table_name = get_table_by_field(request.field_name)
        if not table_name:
            return ArticleQueryResponse(
                success=False,
                message=f"未知的字段类型: {request.field_name}",
                data=[],
                total=0,
                page=request.page,
                page_size=request.page_size
            )
        
        # 获取数据源
        ds_list = get_datasource_list(session, current_user)
        if not ds_list:
            return ArticleQueryResponse(
                success=False,
                message="未找到可用的数据源",
                data=[],
                total=0,
                page=request.page,
                page_size=request.page_size
            )
        
        # 使用指定的数据源或第一个数据源
        ds = None
        if request.datasource_id:
            for d in ds_list:
                if d.id == request.datasource_id:
                    ds = d
                    break
        if not ds:
            ds = ds_list[0]
        
        # 计算分页偏移量
        offset = (request.page - 1) * request.page_size

        # 通用下钻：从原始 SQL 自动提取额外 JOIN 和 WHERE 条件
        normalized_fn = normalize_field_name(request.field_name)
        drill_info = extract_drill_conditions(request.original_sql, normalized_fn)

        print(f"\n{'='*60}")
        print(f"[文章下钻] field_name={request.field_name!r}  field_value={request.field_value!r}")
        print(f"[文章下钻] original_sql={request.original_sql!r}")
        print(f"[文章下钻] 提取额外 JOIN  : {drill_info['extra_joins']}")
        print(f"[文章下钻] 提取额外条件  : {drill_info['extra_conditions']}")

        # 构建查询 SQL
        query_sql = build_query_sql(
            table_name,
            request.field_name,
            request.field_value,
            offset,
            request.page_size,
            extra_joins=drill_info['extra_joins'],
            extra_conditions=drill_info['extra_conditions'],
        )

        if not query_sql:
            return ArticleQueryResponse(
                success=False,
                message="无法构建查询 SQL",
                data=[],
                total=0,
                page=request.page,
                page_size=request.page_size
            )

        # 构建计数 SQL
        count_sql = build_count_sql(
            table_name, request.field_name, request.field_value,
            extra_joins=drill_info['extra_joins'],
            extra_conditions=drill_info['extra_conditions'],
        )

        print(f"[文章下钻] 查询 SQL:\n{query_sql}")
        print(f"[文章下钻] 计数 SQL:\n{count_sql}")
        print(f"{'='*60}\n")
        
        # 执行查询
        result = exec_sql(ds, query_sql, origin_column=True)
        count_result = exec_sql(ds, count_sql, origin_column=True)
        
        # 解析结果
        articles = []
        if result and 'data' in result:
            for row in result['data']:
                article = ArticleInfo(
                    article_title=row.get('article_title'),
                    publish_time=str(row.get('publish_time')) if row.get('publish_time') else None,
                    view_count=int(row.get('view_count')) if row.get('view_count') else 0,
                    article_url=row.get('article_url'),
                    likes=int(row.get('likes')) if row.get('likes') else 0,
                    unit_name=row.get('unit_name'),
                    unit_property=row.get('unit_property'),
                )
                articles.append(article)
        
        # 获取总数
        total = 0
        if count_result and 'data' in count_result and len(count_result['data']) > 0:
            total = int(count_result['data'][0].get('total', 0))
        
        return ArticleQueryResponse(
            success=True,
            message="查询成功",
            data=articles,
            total=total,
            page=request.page,
            page_size=request.page_size
        )
        
    except Exception as e:
        return ArticleQueryResponse(
            success=False,
            message=f"查询失败: {str(e)}",
            data=[],
            total=0,
            page=request.page,
            page_size=request.page_size
        )
