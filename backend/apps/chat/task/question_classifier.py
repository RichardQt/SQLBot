import re

_ACTIVITY_QUERY_KEYWORDS = (
    "活动开展情况",
    "活动开展",
    "活动情况",
    "开展情况",
    "开展得怎么样",
)

_ACTIVITY_INTENT_WORDS = (
    "情况",
    "怎么样",
    "如何",
    "进展",
    "成效",
)


def is_activity_query(question: str) -> bool:
    """判断是否为“活动开展情况”查询。"""
    if not question:
        return False

    normalized = re.sub(r"\s+", "", question)
    if "活动" not in normalized:
        return False

    if any(keyword in normalized for keyword in _ACTIVITY_QUERY_KEYWORDS):
        return True

    # 兼容多轮补全后语序变化，如“开展普法活动的情况”
    if "活动" in normalized and any(word in normalized for word in _ACTIVITY_INTENT_WORDS):
        if re.search(r"开展.{0,8}活动|活动.{0,8}开展", normalized):
            return True

    return False


def extract_unit_name(question: str) -> str:
    """从问句中提取单位名称，提取失败返回“该单位”。"""
    if not question:
        return "该单位"

    normalized = question.strip()

    patterns = [
        r"(?:查询|请问|帮我查(?:询)?)?\s*([^，。？?\s]{1,30}?单位)的?(?:活动开展情况|活动情况|活动开展得怎么样|开展情况)",
        r"(?:查询|请问|帮我查(?:询)?)?\s*([^，。？?\s]{1,30}?单位)(?:活动开展情况|活动情况|活动开展得怎么样|开展情况)",
        r"(?:查询|请问|帮我查(?:询)?)?\s*([^，。？?\s]{1,30}?)(?:的)?活动(?:开展情况|情况|开展得怎么样)",
    ]

    for pattern in patterns:
        matched = re.search(pattern, normalized)
        if matched:
            name = matched.group(1).strip()
            return name if name else "该单位"

    fallback = re.search(r"([^，。？?\s]{1,30}?单位)", normalized)
    if fallback:
        return fallback.group(1).strip()

    return "该单位"
