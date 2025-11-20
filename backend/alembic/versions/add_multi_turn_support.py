"""add multi turn support

Revision ID: e094b46d6771
Revises: a8f3b9e4d1c2
Create Date: 2025-11-20 11:47:42.259136

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'e094b46d6771'
down_revision = 'a8f3b9e4d1c2'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('chat', sa.Column('enable_multi_turn', sa.Boolean(), nullable=False, server_default=sa.text('false')))
    op.add_column('chat_record', sa.Column('complete_question', sa.Text(), nullable=True))
    
    # Data migration: set complete_question = question for existing records
    op.execute("UPDATE chat_record SET complete_question = question")


def downgrade():
    op.drop_column('chat_record', 'complete_question')
    op.drop_column('chat', 'enable_multi_turn')

