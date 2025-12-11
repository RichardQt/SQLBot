"""推荐问题池 CRUD 操作"""
import random
from datetime import datetime
from typing import List

from sqlalchemy import and_, select, func, delete

from apps.chat.models.question_pool_model import RecommendedQuestionPool
from common.core.deps import SessionDep, CurrentUser


def get_random_questions(session: SessionDep, datasource_id: int, oid: int, count: int = 4) -> List[str]:
    """从问题池中随机获取指定数量的问题"""
    # 获取该数据源下所有活跃的问题
    stmt = select(RecommendedQuestionPool.question).where(
        and_(
            RecommendedQuestionPool.datasource_id == datasource_id,
            RecommendedQuestionPool.oid == oid,
            RecommendedQuestionPool.is_active == True
        )
    )
    result = session.execute(stmt).scalars().all()
    
    if not result:
        return []
    
    # 随机选取指定数量的问题
    questions = list(result)
    if len(questions) <= count:
        return questions
    
    return random.sample(questions, count)


def get_question_pool_count(session: SessionDep, datasource_id: int, oid: int) -> int:
    """获取问题池中的问题数量"""
    stmt = select(func.count(RecommendedQuestionPool.id)).where(
        and_(
            RecommendedQuestionPool.datasource_id == datasource_id,
            RecommendedQuestionPool.oid == oid,
            RecommendedQuestionPool.is_active == True
        )
    )
    return session.execute(stmt).scalar() or 0


def save_questions_to_pool(
    session: SessionDep, 
    current_user: CurrentUser, 
    datasource_id: int, 
    questions: List[str]
) -> int:
    """批量保存问题到问题池"""
    saved_count = 0
    now = datetime.now()
    oid = current_user.oid if current_user.oid is not None else 1
    
    for question in questions:
        if not question or not question.strip():
            continue
        
        # 检查是否已存在相同问题
        stmt = select(RecommendedQuestionPool.id).where(
            and_(
                RecommendedQuestionPool.datasource_id == datasource_id,
                RecommendedQuestionPool.oid == oid,
                RecommendedQuestionPool.question == question.strip()
            )
        )
        existing = session.execute(stmt).scalar()
        
        if not existing:
            pool_item = RecommendedQuestionPool(
                oid=oid,
                datasource_id=datasource_id,
                question=question.strip(),
                create_time=now,
                create_by=current_user.id,
                is_active=True
            )
            session.add(pool_item)
            saved_count += 1
    
    session.commit()
    return saved_count


def clear_question_pool(session: SessionDep, datasource_id: int, oid: int) -> int:
    """清空指定数据源的问题池"""
    stmt = delete(RecommendedQuestionPool).where(
        and_(
            RecommendedQuestionPool.datasource_id == datasource_id,
            RecommendedQuestionPool.oid == oid
        )
    )
    result = session.execute(stmt)
    session.commit()
    return result.rowcount


def list_question_pool(
    session: SessionDep, 
    datasource_id: int, 
    oid: int,
    page: int = 1,
    page_size: int = 20
) -> tuple[List[RecommendedQuestionPool], int]:
    """分页获取问题池列表"""
    # 获取总数
    count_stmt = select(func.count(RecommendedQuestionPool.id)).where(
        and_(
            RecommendedQuestionPool.datasource_id == datasource_id,
            RecommendedQuestionPool.oid == oid,
            RecommendedQuestionPool.is_active == True
        )
    )
    total = session.execute(count_stmt).scalar() or 0
    
    # 分页查询
    offset = (page - 1) * page_size
    stmt = select(RecommendedQuestionPool).where(
        and_(
            RecommendedQuestionPool.datasource_id == datasource_id,
            RecommendedQuestionPool.oid == oid,
            RecommendedQuestionPool.is_active == True
        )
    ).order_by(RecommendedQuestionPool.create_time.desc()).offset(offset).limit(page_size)
    
    result = session.execute(stmt).scalars().all()
    return list(result), total
