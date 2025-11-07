"""048_add_chat_log_feedback

Revision ID: a8f3b9e4d1c2
Revises: c1b794a961ce
Create Date: 2025-11-07 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a8f3b9e4d1c2'
down_revision = 'c1b794a961ce'
branch_labels = None
depends_on = None


def upgrade():
    # Add feedback column to chat_log table
    op.add_column('chat_log', sa.Column('feedback', sa.String(length=10), nullable=True))


def downgrade():
    # Remove feedback column from chat_log table
    op.drop_column('chat_log', 'feedback')
