from sqlalchemy import text
from sqlmodel import Session, create_engine, SQLModel

from common.core.config import settings

engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI),
                       pool_size=settings.PG_POOL_SIZE,
                       max_overflow=settings.PG_MAX_OVERFLOW,
                       pool_recycle=settings.PG_POOL_RECYCLE,
                       pool_pre_ping=settings.PG_POOL_PRE_PING)


def get_session():
    with Session(engine) as session:
        yield session


def sync_chat_log_id_sequence():
    with engine.begin() as conn:
        conn.execute(text("""
            SELECT setval(
                pg_get_serial_sequence('chat_log', 'id'),
                COALESCE((SELECT MAX(id) FROM chat_log), 0) + 1,
                false
            )
        """))


def init_db():
    SQLModel.metadata.create_all(engine)
