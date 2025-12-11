"""推荐问题池 API 接口"""
import traceback

import orjson
from fastapi import APIRouter, HTTPException

from apps.chat.curd.question_pool import (
    get_random_questions, 
    get_question_pool_count, 
    clear_question_pool,
    list_question_pool
)
from apps.chat.models.question_pool_model import GenerateQuestionPoolRequest, GenerateQuestionPoolResponse
from apps.chat.task.question_pool_service import QuestionPoolService
from common.core.deps import SessionDep, CurrentUser

router = APIRouter(tags=["Question Pool"], prefix="/question_pool")


@router.post("/generate", response_model=GenerateQuestionPoolResponse)
async def generate_question_pool(
    session: SessionDep, 
    current_user: CurrentUser, 
    request: GenerateQuestionPoolRequest
):
    """
    批量生成推荐问题并存入问题池
    
    - datasource_id: 数据源ID
    - count: 需要生成的问题数量，默认100
    """
    try:
        service = QuestionPoolService(session, current_user, request.datasource_id)
        result = await service.generate_batch_questions(request.count)
        return GenerateQuestionPoolResponse(**result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/random/{datasource_id}")
async def get_random_pool_questions(
    session: SessionDep, 
    current_user: CurrentUser, 
    datasource_id: int,
    count: int = 4
):
    """
    从问题池中随机获取问题
    
    - datasource_id: 数据源ID
    - count: 获取的问题数量，默认4
    """
    try:
        oid = current_user.oid if current_user.oid is not None else 1
        questions = get_random_questions(session, datasource_id, oid, count)
        return {
            'datasource_id': datasource_id,
            'questions': questions,
            'count': len(questions)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/count/{datasource_id}")
async def get_pool_count(
    session: SessionDep, 
    current_user: CurrentUser, 
    datasource_id: int
):
    """获取问题池中的问题数量"""
    try:
        oid = current_user.oid if current_user.oid is not None else 1
        count = get_question_pool_count(session, datasource_id, oid)
        return {
            'datasource_id': datasource_id,
            'count': count
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear/{datasource_id}")
async def clear_pool(
    session: SessionDep, 
    current_user: CurrentUser, 
    datasource_id: int
):
    """清空指定数据源的问题池"""
    try:
        oid = current_user.oid if current_user.oid is not None else 1
        deleted_count = clear_question_pool(session, datasource_id, oid)
        return {
            'datasource_id': datasource_id,
            'deleted_count': deleted_count
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list/{datasource_id}")
async def list_pool_questions(
    session: SessionDep, 
    current_user: CurrentUser, 
    datasource_id: int,
    page: int = 1,
    page_size: int = 20
):
    """分页获取问题池列表"""
    try:
        oid = current_user.oid if current_user.oid is not None else 1
        questions, total = list_question_pool(session, datasource_id, oid, page, page_size)
        return {
            'datasource_id': datasource_id,
            'questions': [{'id': q.id, 'question': q.question, 'create_time': q.create_time} for q in questions],
            'total': total,
            'page': page,
            'page_size': page_size
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
