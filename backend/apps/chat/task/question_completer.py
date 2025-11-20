from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from common.utils.utils import extract_nested_json
from apps.template.multi_turn.generator import get_question_complete_template
import orjson

class QuestionCompleter:
    """负责生成完整问题"""
    
    def __init__(self, llm: BaseChatModel, lang: str):
        self.llm = llm
        self.lang = lang
    
    def complete_question(
        self, 
        previous_question: str, 
        current_question: str
    ) -> str:
        """
        使用大模型生成包含上下文的完整问题
        
        Args:
            previous_question: 上一轮问题
            current_question: 当前问题
            
        Returns:
            str: 完整问题
        """
        template = get_question_complete_template()
        system_prompt = template['system'].format(
            previous_question=previous_question,
            current_question=current_question
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Rewrite the question.")
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content
            
            json_str = extract_nested_json(content)
            if json_str:
                data = orjson.loads(json_str)
                return data.get("complete_question", current_question)
        except Exception:
            pass
            
        return current_question
