"""create recommended_question_pool table

Revision ID: f1a2b3c4d5e6
Revises: e094b46d6771
Create Date: 2024-12-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'e094b46d6771'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'recommended_question_pool',
        sa.Column('id', sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        sa.Column('oid', sa.BigInteger(), nullable=False, server_default='1'),
        sa.Column('datasource_id', sa.BigInteger(), nullable=False, index=True),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('create_time', sa.DateTime(), nullable=True),
        sa.Column('create_by', sa.BigInteger(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
    )
    
    # 创建索引以加速按数据源查询
    op.create_index(
        'ix_recommended_question_pool_datasource_oid',
        'recommended_question_pool',
        ['datasource_id', 'oid']
    )


def downgrade() -> None:
    op.drop_index('ix_recommended_question_pool_datasource_oid', table_name='recommended_question_pool')
    op.drop_table('recommended_question_pool')
