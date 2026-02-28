"""
图表点击事件 - 文章详情查询 API
根据图表点击数据查询相关文章的详细信息
"""
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

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


def build_query_sql(table_name: str, field_name: str, field_value: str, offset: int, limit: int) -> str:
    """
    根据表名和字段构建查询 SQL
    返回文章详情：article_title, publish_time, view_count, article_url, likes
    """
    # 先规范化字段名
    field_name = normalize_field_name(field_name)
    # 特殊处理主题日字段
    if field_name == 'theme_name':
        field_name = 'Legal_topics'
    # 特殊处理区域字段：图表中是 district_name，数据库中是 unit_district
    if field_name == 'district_name':
        field_name = 'unit_district'
    
    # 根据不同的表构建不同的查询
    if table_name == 'fx_education_articles':
        # 单位名称和单位属性直接从 fx_education_articles 关联
        sql = f"""
            SELECT DISTINCT
                r.article_title,
                r.publish_time,
                r.view_count,
                r.article_url,
                r.thumbs_count AS likes
            FROM fx_education_articles e
            JOIN fx_article_records r ON e.article_id = r.article_id
            WHERE e.type_class = '1'
              AND e.{field_name} = '{field_value}'
            ORDER BY r.publish_time DESC
            LIMIT {limit} OFFSET {offset}
        """
    elif table_name == 'fx_education_articles_legal':
        # 法律法规表
        sql = f"""
            SELECT DISTINCT
                r.article_title,
                r.publish_time,
                r.view_count,
                r.article_url,
                r.thumbs_count AS likes
            FROM fx_education_articles_legal l
            JOIN fx_education_articles e ON l.article_id = e.article_id
            
            JOIN fx_article_records r ON e.article_id = r.article_id

            WHERE e.type_class = '1'
              AND l.{field_name} = '{field_value}'
            ORDER BY r.publish_time DESC
            LIMIT {limit} OFFSET {offset}
        """
    elif table_name == 'fx_education_articles_group':
        # 受众群体表
        sql = f"""
            SELECT DISTINCT
                r.article_title,
                r.publish_time,
                r.view_count,
                r.article_url,
                r.thumbs_count AS likes
            FROM fx_education_articles_group g
            JOIN fx_education_articles e ON g.article_id = e.article_id
            JOIN fx_article_records r ON e.article_id = r.article_id
            WHERE e.type_class = '1'
              AND g.{field_name} = '{field_value}'
            ORDER BY r.publish_time DESC
            LIMIT {limit} OFFSET {offset}
        """
    else:
        return None
    
    return sql


def build_count_sql(table_name: str, field_name: str, field_value: str) -> str:
    """构建计数 SQL"""
    # 先规范化字段名
    field_name = normalize_field_name(field_name)
    # 特殊处理主题日字段
    if field_name == 'theme_name':
        field_name = 'Legal_topics'
    # 特殊处理区域字段：图表中是 district_name，数据库中是 unit_district
    if field_name == 'district_name':
        field_name = 'unit_district'
    
    if table_name == 'fx_education_articles':
        sql = f"""
            SELECT COUNT(DISTINCT r.article_id) AS total
            FROM fx_education_articles e
            JOIN fx_article_records r ON e.article_id = r.article_id
            WHERE e.type_class = '1'
              AND e.{field_name} = '{field_value}'
        """
    elif table_name == 'fx_education_articles_legal':
        sql = f"""
            SELECT COUNT(DISTINCT r.article_id) AS total
            FROM fx_education_articles_legal l
            JOIN fx_education_articles e ON l.article_id = e.article_id
            JOIN fx_article_records r ON e.article_id = r.article_id
            WHERE e.type_class = '1'
              AND l.{field_name} = '{field_value}'
        """
    elif table_name == 'fx_education_articles_group':
        sql = f"""
            SELECT COUNT(DISTINCT r.article_id) AS total
            FROM fx_education_articles_group g
            JOIN fx_education_articles e ON g.article_id = e.article_id
            JOIN fx_article_records r ON e.article_id = r.article_id
            WHERE e.type_class = '1'
              AND g.{field_name} = '{field_value}'
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
        
        # 构建查询 SQL
        query_sql = build_query_sql(
            table_name, 
            request.field_name, 
            request.field_value,
            offset,
            request.page_size
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
        count_sql = build_count_sql(table_name, request.field_name, request.field_value)
        
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
                    likes=int(row.get('likes')) if row.get('likes') else 0
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
