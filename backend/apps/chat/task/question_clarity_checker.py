import orjson
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from apps.template.multi_turn.generator import get_question_clarity_template
from common.utils.utils import extract_nested_json


class QuestionClarityChecker:
    """负责检查问题是否足够明确"""

    def __init__(self, llm: BaseChatModel, lang: str):
        self.llm = llm
        self.lang = lang

    def check_clarity(
        self,
        question: str,
        db_schema: str
    ) -> tuple[bool, str]:
        """
        检查问题是否足够明确，能够独立生成SQL

        Args:
            question: 用户问题
            db_schema: 数据库Schema

        Returns:
            tuple[bool, str]: (是否明确, 提示信息)
        """
        template = get_question_clarity_template()
        format_kwargs = {
            'question': question,
            'schema': db_schema,
            'lang': self.lang,
        }

        system_prompt_tpl = template.get('system')
        if not system_prompt_tpl:
            raise ValueError('question_clarity.system 模板缺失')

        messages = [SystemMessage(content=system_prompt_tpl.format(**format_kwargs))]

        user_prompt_tpl = template.get('user')
        if user_prompt_tpl:
            messages.append(HumanMessage(content=user_prompt_tpl.format(**format_kwargs)))

        try:
            response = self.llm.invoke(messages)
            content = response.content

            json_str = extract_nested_json(content)
            if json_str:
                data = orjson.loads(json_str)
                is_clear = data.get("is_clear", True)
                suggestion = data.get("suggestion", "")
                return is_clear, suggestion
        except Exception:
            pass

        # Default to clear if parsing fails to avoid blocking user
        return True, ""
