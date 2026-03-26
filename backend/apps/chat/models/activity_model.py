from typing import Optional

from sqlmodel import Field, SQLModel


class FxEducationArticlesKeyword(SQLModel, table=True):
    """活动开展情况表模型（仅声明核心字段，供类型提示与后续扩展）。"""

    __tablename__ = "fx_education_articles_keyword"

    id: Optional[int] = Field(default=None, primary_key=True)
    unit_name: Optional[str] = Field(default=None)
    service_logic: Optional[str] = Field(default=None)
    result_value: Optional[str] = Field(default=None)
