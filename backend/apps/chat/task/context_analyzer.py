from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from common.utils.utils import extract_nested_json
from apps.template.multi_turn.generator import get_context_analysis_template
import orjson

class ContextAnalyzer:
    """负责分析问题之间的关联性"""
    
    def __init__(self, llm: BaseChatModel, lang: str):
        self.llm = llm
        self.lang = lang
    
    def analyze_relevance(
        self, 
        previous_question: str, 
        current_question: str
    ) -> bool:
        """
        使用大模型判断两个问题是否存在关联
        
        Args:
            previous_question: 上一轮问题
            current_question: 当前问题
            
        Returns:
            bool: True表示存在关联，False表示不存在关联
        """
        template = get_context_analysis_template()
        system_prompt = template['system'].format(
            previous_question=previous_question,
            current_question=current_question
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Analyze the relevance.")
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content
            
            json_str = extract_nested_json(content)
            if json_str:
                data = orjson.loads(json_str)
                return data.get("is_related", False)
        except Exception:
            pass
            
        return False
