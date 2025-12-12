from __future__ import annotations

import datetime

from typing import Optional

from fastapi import APIRouter, Query
from sqlalchemy import BigInteger, and_, func
from sqlmodel import delete as sqlmodel_delete
from sqlmodel import select

from apps.system.models.custom_prompt_model import CustomPrompt, CustomPromptInfo
from apps.datasource.models.datasource import CoreDatasource
from common.core.deps import CurrentUser, SessionDep, Trans


router = APIRouter(tags=["system/custom_prompt"], prefix="/system/custom_prompt")


@router.get("/{prompt_type}/page/{current_page}/{page_size}")
async def page_custom_prompt(
    session: SessionDep,
    current_user: CurrentUser,
    prompt_type: str,
    current_page: int,
    page_size: int,
    name: Optional[str] = Query(None, description="提示词名称(可选)"),
):
    """分页获取自定义提示词列表。

    前端调用：GET `/system/custom_prompt/${type}/page/${pageNum}/${pageSize}`
    返回结构对齐前端使用：`res.data` 与 `res.total_count`。
    """

    oid = current_user.oid

    current_page = max(1, current_page)
    page_size = max(1, page_size)

    keyword = name.strip() if name else None
    keyword_pattern = f"%{keyword}%" if keyword else None

    base_ids_stmt = select(CustomPrompt.id).where(
        and_(
            CustomPrompt.oid == oid,
            CustomPrompt.type == prompt_type,
        )
    )
    if keyword_pattern:
        base_ids_stmt = base_ids_stmt.where(CustomPrompt.name.ilike(keyword_pattern))

    count_stmt = select(func.count()).select_from(base_ids_stmt.subquery())
    total_count = session.exec(count_stmt).first() or 0
    total_pages = (total_count + page_size - 1) // page_size if total_count else 0

    if total_pages and current_page > total_pages:
        current_page = 1

    paginated_ids = (
        base_ids_stmt.order_by(CustomPrompt.create_time.desc())
        .offset((current_page - 1) * page_size)
        .limit(page_size)
        .subquery()
    )

    datasource_names_subquery = (
        select(
            func.jsonb_array_elements(CustomPrompt.datasource_ids).cast(BigInteger).label("ds_id"),
            CustomPrompt.id.label("cp_id"),
        )
        .where(CustomPrompt.id.in_(paginated_ids))
        .subquery()
    )

    stmt = (
        select(
            CustomPrompt.id,
            CustomPrompt.type,
            CustomPrompt.name,
            CustomPrompt.prompt,
            CustomPrompt.create_time,
            CustomPrompt.specific_ds,
            CustomPrompt.datasource_ids,
            func.jsonb_agg(CoreDatasource.name)
            .filter(CoreDatasource.id.isnot(None))
            .label("datasource_names"),
        )
        .outerjoin(
            datasource_names_subquery,
            datasource_names_subquery.c.cp_id == CustomPrompt.id,
        )
        .outerjoin(CoreDatasource, CoreDatasource.id == datasource_names_subquery.c.ds_id)
        .where(CustomPrompt.id.in_(paginated_ids))
        .group_by(
            CustomPrompt.id,
            CustomPrompt.type,
            CustomPrompt.name,
            CustomPrompt.prompt,
            CustomPrompt.create_time,
            CustomPrompt.specific_ds,
            CustomPrompt.datasource_ids,
        )
        .order_by(CustomPrompt.create_time.desc())
    )

    result = session.exec(stmt).all()
    data: list[dict] = []
    for row in result:
        # row 是 Row/tuple 风格对象，使用属性访问
        data.append(
            {
                "id": row.id,
                "type": row.type,
                "name": row.name,
                "prompt": row.prompt,
                "create_time": row.create_time,
                "specific_ds": bool(row.specific_ds) if row.specific_ds is not None else False,
                "datasource_ids": row.datasource_ids if row.datasource_ids is not None else [],
                "datasource_names": row.datasource_names if row.datasource_names is not None else [],
            }
        )

    return {
        "current_page": current_page,
        "page_size": page_size,
        "total_count": total_count,
        "total_pages": total_pages,
        "data": data,
    }


@router.get("/{id}")
async def get_custom_prompt(session: SessionDep, current_user: CurrentUser, id: int):
    """获取单条提示词详情（用于编辑回填）。"""

    oid = current_user.oid
    db_model = session.get(CustomPrompt, id)
    if not db_model:
        raise Exception("提示词不存在")
    if getattr(db_model, "oid", None) != oid:
        raise Exception("无权限查看该提示词")

    return db_model


@router.delete("")
async def delete_custom_prompt(session: SessionDep, current_user: CurrentUser, id_list: list[int]):
    """批量删除提示词。

    前端调用：DELETE `/system/custom_prompt`，body 为 id 数组。
    """

    if not id_list:
        return 0

    oid = current_user.oid
    stmt = sqlmodel_delete(CustomPrompt).where(
        and_(
            CustomPrompt.oid == oid,
            CustomPrompt.id.in_(id_list),
        )
    )
    result = session.exec(stmt)
    session.commit()
    return result.rowcount or 0


@router.put("")
async def create_or_update_custom_prompt(
    session: SessionDep,
    current_user: CurrentUser,
    trans: Trans,
    info: CustomPromptInfo,
):
    """创建或更新自定义提示词。

    说明：
    - 该接口用于解决前端调用 PUT `/system/custom_prompt` 时出现 405 的问题。
    - GET/DELETE 等接口可能由 xpack 提供；这里仅补齐 PUT，避免重复注册导致启动失败。
    """

    if not info.type:
        raise Exception(trans("i18n_miss_args", key="type"))
    if not info.name or not str(info.name).strip():
        raise Exception(trans("i18n_miss_args", key="name"))
    if not info.prompt or not str(info.prompt).strip():
        raise Exception(trans("i18n_miss_args", key="prompt"))

    specific_ds = bool(info.specific_ds)
    datasource_ids = info.datasource_ids or []
    if specific_ds and not datasource_ids:
        raise Exception(trans("i18n_miss_args", key="datasource_ids"))
    if not specific_ds:
        datasource_ids = []

    oid = current_user.oid

    # 简单去重：同一工作空间(oid) + type + name 不允许重复
    dup_stmt = select(CustomPrompt.id).where(
        and_(
            CustomPrompt.oid == oid,
            CustomPrompt.type == info.type,
            CustomPrompt.name == info.name,
            CustomPrompt.id != info.id if info.id else True,
        )
    )
    dup_id = session.exec(dup_stmt).first()
    if dup_id is not None:
        raise Exception("提示词名称已存在，请换一个名称")

    if info.id:
        db_model = session.get(CustomPrompt, int(info.id))
        if not db_model:
            raise Exception("提示词不存在")
        if db_model.oid != oid:
            raise Exception("无权限操作该提示词")

        db_model.type = info.type
        db_model.name = str(info.name).strip()
        db_model.prompt = info.prompt
        db_model.specific_ds = specific_ds
        db_model.datasource_ids = datasource_ids
        session.add(db_model)
        session.commit()
        return db_model.id

    db_model = CustomPrompt(
        oid=oid,
        type=info.type,
        create_time=datetime.datetime.now(),
        name=str(info.name).strip(),
        prompt=info.prompt,
        specific_ds=specific_ds,
        datasource_ids=datasource_ids,
    )
    session.add(db_model)
    session.commit()
    session.refresh(db_model)
    return db_model.id
