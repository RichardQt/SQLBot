"""推荐问题池模型 - 用于存储批量预生成的推荐问题"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Column, BigInteger, DateTime, Identity, Text, Boolean
from sqlmodel import SQLModel, Field


class RecommendedQuestionPool(SQLModel, table=True):
    """推荐问题池表 - 按数据源存储预生成的问题"""
    __tablename__ = "recommended_question_pool"
    
    id: Optional[int] = Field(sa_column=Column(BigInteger, Identity(always=True), primary_key=True))
    oid: int = Field(sa_column=Column(BigInteger, nullable=False, default=1))
    datasource_id: int = Field(sa_column=Column(BigInteger, nullable=False, index=True))
    question: str = Field(sa_column=Column(Text, nullable=False))
    create_time: datetime = Field(sa_column=Column(DateTime(timezone=False), nullable=True))
    create_by: int = Field(sa_column=Column(BigInteger, nullable=True))
    is_active: bool = Field(sa_column=Column(Boolean, nullable=False, default=True))


class GenerateQuestionPoolRequest(BaseModel):
    """批量生成问题请求"""
    datasource_id: int
    count: int = 100  # 默认生成100个


class GenerateQuestionPoolResponse(BaseModel):
    """批量生成问题响应"""
    datasource_id: int
    generated_count: int
    total_count: int
