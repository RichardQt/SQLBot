"""merge heads

Revision ID: 1037b1f32607
Revises: 5755c0b95839, e094b46d6771
Create Date: 2025-12-05 16:00:39.687505

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = '1037b1f32607'
down_revision = ('5755c0b95839', 'e094b46d6771')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
