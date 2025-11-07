#!/usr/bin/env python3
"""
简化版重新生成所有 embedding 的脚本
直接使用项目的数据库配置

使用方法:
cd /opt/sqlbot/app
python scripts/regenerate_embeddings_simple.py
"""
import traceback
from sqlalchemy import update
from sqlalchemy.orm import sessionmaker, scoped_session

from apps.datasource.models.datasource import CoreTable, CoreDatasource
from apps.datasource.crud.table import save_table_embedding, save_ds_embedding
from common.core.config import settings
from common.core.db import engine
from common.utils.utils import SQLBotLogUtil

# 创建 session_maker
session_maker = scoped_session(sessionmaker(bind=engine))


def clear_all_embeddings():
    """清空所有 embedding 数据"""
    try:
        SQLBotLogUtil.info('开始清空所有 embedding 数据...')
        session = session_maker()
        
        # 清空表的 embedding
        table_stmt = update(CoreTable).values(embedding=None)
        result = session.execute(table_stmt)
        table_count = result.rowcount
        SQLBotLogUtil.info(f'已清空 {table_count} 个表的 embedding')
        
        # 清空数据源的 embedding
        ds_stmt = update(CoreDatasource).values(embedding=None)
        result = session.execute(ds_stmt)
        ds_count = result.rowcount
        SQLBotLogUtil.info(f'已清空 {ds_count} 个数据源的 embedding')
        
        session.commit()
        SQLBotLogUtil.info('所有 embedding 数据已清空')
        return table_count, ds_count
    except Exception as e:
        SQLBotLogUtil.error(f'清空 embedding 失败: {str(e)}')
        traceback.print_exc()
        session.rollback()
        raise
    finally:
        session.close()


def regenerate_all_embeddings():
    """重新生成所有 embedding"""
    try:
        if not settings.TABLE_EMBEDDING_ENABLED:
            SQLBotLogUtil.warning('TABLE_EMBEDDING_ENABLED 未启用')
            return
        
        SQLBotLogUtil.info('开始重新生成所有 embedding...')
        session = session_maker()
        
        # 获取所有表 ID
        SQLBotLogUtil.info('获取所有表...')
        table_ids = [t.id for t in session.query(CoreTable.id).all()]
        SQLBotLogUtil.info(f'找到 {len(table_ids)} 个表需要生成 embedding')
        
        # 获取所有数据源 ID
        SQLBotLogUtil.info('获取所有数据源...')
        ds_ids = [ds.id for ds in session.query(CoreDatasource.id).all()]
        SQLBotLogUtil.info(f'找到 {len(ds_ids)} 个数据源需要生成 embedding')
        
        session.close()
        
        # 生成表的 embedding
        if table_ids:
            SQLBotLogUtil.info('正在生成表的 embedding...')
            save_table_embedding(session_maker, table_ids)
            SQLBotLogUtil.info('表的 embedding 生成完成')
        
        # 生成数据源的 embedding
        if ds_ids:
            SQLBotLogUtil.info('正在生成数据源的 embedding...')
            save_ds_embedding(session_maker, ds_ids)
            SQLBotLogUtil.info('数据源的 embedding 生成完成')
        
        SQLBotLogUtil.info('所有 embedding 重新生成完成!')
        
    except Exception as e:
        SQLBotLogUtil.error(f'重新生成 embedding 失败: {str(e)}')
        traceback.print_exc()
        raise
    finally:
        session_maker.remove()


def main():
    """主函数"""
    try:
        print('=' * 60)
        print('开始重新生成 embedding 任务')
        print(f'当前使用的 embedding 模型: {settings.DEFAULT_EMBEDDING_MODEL}')
        print('=' * 60)
        
        SQLBotLogUtil.info('=' * 60)
        SQLBotLogUtil.info('开始重新生成 embedding 任务')
        SQLBotLogUtil.info(f'当前使用的 embedding 模型: {settings.DEFAULT_EMBEDDING_MODEL}')
        SQLBotLogUtil.info('=' * 60)
        
        # 清空所有 embedding
        table_count, ds_count = clear_all_embeddings()
        
        print('\n' + '=' * 60)
        print(f'已清空 {table_count} 个表和 {ds_count} 个数据源的 embedding')
        print('=' * 60)
        
        # 确认是否继续
        response = input('\n是否继续重新生成所有 embedding? (yes/no): ').strip().lower()
        if response not in ['yes', 'y', '是']:
            SQLBotLogUtil.info('操作已取消')
            print('操作已取消')
            return
        
        # 重新生成所有 embedding
        regenerate_all_embeddings()
        
        print('\n' + '=' * 60)
        print('✓ 所有 embedding 已成功重新生成!')
        print(f'✓ 使用的模型: {settings.DEFAULT_EMBEDDING_MODEL}')
        print('=' * 60)
        
    except Exception as e:
        print('\n' + '=' * 60)
        print(f'✗ 执行失败: {str(e)}')
        print('=' * 60)
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
