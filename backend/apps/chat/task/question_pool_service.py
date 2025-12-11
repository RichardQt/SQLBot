"""推荐问题池批量生成服务"""
import traceback
from typing import List, Any, Union

import orjson
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from apps.ai_model.model_factory import LLMFactory, get_default_config
from apps.chat.curd.question_pool import save_questions_to_pool, get_question_pool_count
from apps.datasource.crud.datasource import get_table_schema
from apps.datasource.models.datasource import CoreDatasource
from apps.template.generate_guess_question.batch_generator import get_guess_batch_template
from common.core.deps import CurrentUser, SessionDep
from common.utils.utils import SQLBotLogUtil, extract_nested_json


def get_lang_name(lang: str) -> str:
    """获取语言名称"""
    lang_map = {
        'zh-CN': '简体中文',
        'en': 'English',
    }
    return lang_map.get(lang, '简体中文')


class QuestionPoolService:
    """推荐问题池服务 - 批量生成问题"""
    
    def __init__(self, session: SessionDep, current_user: CurrentUser, datasource_id: int):
        self.session = session
        self.current_user = current_user
        self.datasource_id = datasource_id
        self.lang = get_lang_name(current_user.language)
        
        # 获取数据源
        self.ds = session.get(CoreDatasource, datasource_id)
        if not self.ds:
            raise Exception(f"Datasource with id {datasource_id} not found")
        
        # 获取表结构
        self.db_schema = get_table_schema(
            session=session, 
            current_user=current_user, 
            ds=self.ds,
            question="",
            embedding=False
        )
    
    async def generate_batch_questions(self, count: int = 100) -> dict:
        """
        批量生成推荐问题并存入数据库
        
        Args:
            count: 需要生成的问题数量，默认100
            
        Returns:
            dict: 包含生成结果的字典
        """
        try:
            # 获取LLM配置
            config = await get_default_config()
            llm_instance = LLMFactory.create_llm(config)
            llm = llm_instance.llm
            
            # 构建消息
            template = get_guess_batch_template()
            messages: List[Union[BaseMessage, dict[str, Any]]] = []
            
            system_content = template['system'].format(lang=self.lang, count=count)
            user_content = template['user'].format(schema=self.db_schema, count=count)
            
            messages.append(SystemMessage(content=system_content))
            messages.append(HumanMessage(content=user_content))
            
            SQLBotLogUtil.info(f"Starting batch question generation for datasource {self.datasource_id}, count={count}")
            SQLBotLogUtil.info(f"System prompt length: {len(system_content)}, User prompt length: {len(user_content)}")
            SQLBotLogUtil.info(f"DB Schema preview: {self.db_schema[:500] if self.db_schema else 'EMPTY'}")
            
            # 调用LLM生成问题
            full_text = ''
            for chunk in llm.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    full_text += chunk.content
            
            SQLBotLogUtil.info(f"LLM response length: {len(full_text)}, preview: {full_text[:500] if full_text else 'EMPTY'}")
            
            # 解析JSON结果
            questions = self._parse_questions(full_text)
            
            if not questions:
                SQLBotLogUtil.warning(f"No questions generated for datasource {self.datasource_id}")
                return {
                    'datasource_id': self.datasource_id,
                    'generated_count': 0,
                    'total_count': get_question_pool_count(
                        self.session, 
                        self.datasource_id, 
                        self.current_user.oid or 1
                    )
                }
            
            # 保存到数据库
            saved_count = save_questions_to_pool(
                self.session,
                self.current_user,
                self.datasource_id,
                questions
            )
            
            total_count = get_question_pool_count(
                self.session, 
                self.datasource_id, 
                self.current_user.oid or 1
            )
            
            SQLBotLogUtil.info(f"Generated {len(questions)} questions, saved {saved_count} new questions for datasource {self.datasource_id}")
            
            return {
                'datasource_id': self.datasource_id,
                'generated_count': saved_count,
                'total_count': total_count
            }
            
        except Exception as e:
            SQLBotLogUtil.error(f"Error generating batch questions: {str(e)}")
            traceback.print_exc()
            raise
    
    def _parse_questions(self, text: str) -> List[str]:
        """解析LLM返回的问题列表"""
        try:
            SQLBotLogUtil.info(f"Parsing questions from text (first 1000 chars): {text[:1000]}")
            
            # 尝试提取JSON
            json_str = extract_nested_json(text)
            SQLBotLogUtil.info(f"Extracted JSON: {json_str[:500] if json_str else 'None'}")
            
            if json_str:
                questions = orjson.loads(json_str)
                SQLBotLogUtil.info(f"Parsed questions count: {len(questions) if isinstance(questions, list) else 'not a list'}")
                if isinstance(questions, list):
                    result = [q for q in questions if isinstance(q, str) and q.strip()]
                    SQLBotLogUtil.info(f"Filtered questions count: {len(result)}")
                    return result
            
            # 直接尝试解析
            questions = orjson.loads(text)
            if isinstance(questions, list):
                return [q for q in questions if isinstance(q, str) and q.strip()]
                
        except Exception as e:
            SQLBotLogUtil.warning(f"Failed to parse questions JSON: {str(e)}")
        
        return []
