#!/usr/bin/env python3
"""
重新生成所有 embedding 的脚本
用于解决向量维度不匹配的问题

使用方法:
python scripts/regenerate_embeddings.py
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import traceback
from sqlalchemy import create_engine, update
from sqlalchemy.orm import sessionmaker, scoped_session

from apps.datasource.models.datasource import CoreTable, CoreDatasource
from apps.datasource.crud.table import save_table_embedding, save_ds_embedding
from apps.terminology.models.terminology_model import Terminology
from apps.terminology.curd.terminology import save_embeddings as save_terminology_embeddings
from apps.data_training.models.data_training_model import DataTraining
from apps.data_training.curd.data_training import save_embeddings as save_training_embeddings
from common.core.config import settings
from common.utils.utils import SQLBotLogUtil


def clear_all_embeddings(session_maker):
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

        # 清空术语的 embedding
        term_stmt = update(Terminology).values(embedding=None)
        result = session.execute(term_stmt)
        term_count = result.rowcount
        SQLBotLogUtil.info(f'已清空 {term_count} 个术语的 embedding')

        # 清空数据训练的 embedding
        train_stmt = update(DataTraining).values(embedding=None)
        result = session.execute(train_stmt)
        train_count = result.rowcount
        SQLBotLogUtil.info(f'已清空 {train_count} 个数据训练的 embedding')
        
        session.commit()
        SQLBotLogUtil.info('所有 embedding 数据已清空')
        return table_count, ds_count, term_count, train_count
    except Exception as e:
        SQLBotLogUtil.error(f'清空 embedding 失败: {str(e)}')
        traceback.print_exc()
        session.rollback()
        raise
    finally:
        session.close()


def regenerate_all_embeddings(session_maker):
    """重新生成所有 embedding"""
    try:
        SQLBotLogUtil.info('开始重新生成所有 embedding...')
        
        if settings.TABLE_EMBEDDING_ENABLED:
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
        else:
            SQLBotLogUtil.warning('TABLE_EMBEDDING_ENABLED 未启用，跳过表和数据源 embedding 生成')

        if settings.EMBEDDING_ENABLED:
            session = session_maker()

            # 获取所有术语 ID
            SQLBotLogUtil.info('获取所有术语...')
            term_ids = [t.id for t in session.query(Terminology.id).all()]
            SQLBotLogUtil.info(f'找到 {len(term_ids)} 个术语需要生成 embedding')

            # 获取所有数据训练 ID
            SQLBotLogUtil.info('获取所有数据训练...')
            train_ids = [t.id for t in session.query(DataTraining.id).all()]
            SQLBotLogUtil.info(f'找到 {len(train_ids)} 个数据训练需要生成 embedding')

            session.close()

            # 生成术语的 embedding
            if term_ids:
                SQLBotLogUtil.info('正在生成术语的 embedding...')
                save_terminology_embeddings(session_maker, term_ids)
                SQLBotLogUtil.info('术语的 embedding 生成完成')

            # 生成数据训练的 embedding
            if train_ids:
                SQLBotLogUtil.info('正在生成数据训练的 embedding...')
                save_training_embeddings(session_maker, train_ids)
                SQLBotLogUtil.info('数据训练的 embedding 生成完成')
        else:
            SQLBotLogUtil.warning('EMBEDDING_ENABLED 未启用，跳过术语和数据训练 embedding 生成')
        
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
        SQLBotLogUtil.info('=' * 60)
        SQLBotLogUtil.info('开始重新生成 embedding 任务')
        SQLBotLogUtil.info(f'当前使用的 embedding 模型: {settings.DEFAULT_EMBEDDING_MODEL}')
        SQLBotLogUtil.info('=' * 60)
        
        # 创建数据库连接
        engine = create_engine(
            str(settings.SQLALCHEMY_DATABASE_URI),
            pool_pre_ping=True,
            echo=False,
        )
        session_factory = sessionmaker(bind=engine)
        session_maker = scoped_session(session_factory)
        
        # 清空所有 embedding
        table_count, ds_count, term_count, train_count = clear_all_embeddings(session_maker)
        
        print('\n' + '=' * 60)
        print(f'已清空 {table_count} 个表, {ds_count} 个数据源, {term_count} 个术语, {train_count} 个数据训练的 embedding')
        print('=' * 60)
        
        # 确认是否继续
        response = input('\n是否继续重新生成所有 embedding? (yes/no): ').strip().lower()
        if response not in ['yes', 'y', '是']:
            SQLBotLogUtil.info('操作已取消')
            print('操作已取消')
            return
        
        # 重新生成所有 embedding
        regenerate_all_embeddings(session_maker)
        
        print('\n' + '=' * 60)
        print('✓ 所有 embedding 已成功重新生成!')
        print(f'✓ 使用的模型: {settings.DEFAULT_EMBEDDING_MODEL}')
        print('=' * 60)
        
    except Exception as e:
        print('\n' + '=' * 60)
        print(f'✗ 执行失败: {str(e)}')
        print('=' * 60)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
