"""修复 Alembic 版本问题的脚本"""
from sqlalchemy import create_engine, text
from common.core.config import settings

def main():
    engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))
    
    with engine.connect() as conn:
        # 查看当前版本
        print("当前数据库中的 Alembic 版本:")
        result = conn.execute(text('SELECT version_num FROM alembic_version'))
        for row in result:
            print(f"  - {row[0]}")
        
        # 删除错误的版本记录
        print("\n正在删除错误的版本记录...")
        conn.execute(text('DELETE FROM alembic_version'))
        conn.commit()
        
        print("✅ 已删除所有版本记录")
        print("现在可以运行 'alembic upgrade head' 来重新初始化数据库迁移")

if __name__ == "__main__":
    main()
