import re

_ACTIVITY_TEMPLATE_NAME = "activity_report.md"
_RESPONSIBILITY_TEMPLATE_NAME = "pufazerenzhi.md"

_ACTIVITY_QUERY_KEYWORDS = (
    "活动开展情况",
    "活动开展",
    "活动情况",
    "开展情况",
    "开展得怎么样",
)

_RESPONSIBILITY_QUERY_KEYWORDS = (
    "普法责任制履职情况",
    "责任制履职情况",
    "责任制履职",
)

_ACTIVITY_INTENT_WORDS = (
    "情况",
    "怎么样",
    "如何",
    "进展",
    "成效",
)

_RESPONSIBILITY_INTENT_WORDS = (
    "履职",
    "落实情况",
    "情况",
    "进展",
    "成效",
)


def _normalize_question(question: str) -> str:
    if not question:
        return ""
    return re.sub(r"\s+", "", question)


def resolve_activity_template(question: str) -> str:
    """根据问句语义选择活动报告模板。

    Returns:
        activity_report.md | pufazerenzhi.md | ""
    """
    normalized = _normalize_question(question)
    if not normalized:
        return ""

    # 优先识别“普法责任制履职情况”类问题
    if any(keyword in normalized for keyword in _RESPONSIBILITY_QUERY_KEYWORDS):
        return _RESPONSIBILITY_TEMPLATE_NAME

    if "责任制" in normalized and any(word in normalized for word in _RESPONSIBILITY_INTENT_WORDS):
        return _RESPONSIBILITY_TEMPLATE_NAME

    # 识别“活动开展情况”类问题
    if any(keyword in normalized for keyword in _ACTIVITY_QUERY_KEYWORDS):
        return _ACTIVITY_TEMPLATE_NAME

    # 兼容语序变化，如“开展普法活动的情况”
    if "活动" in normalized and any(word in normalized for word in _ACTIVITY_INTENT_WORDS):
        if re.search(r"开展.{0,8}活动|活动.{0,8}开展|活动.{0,8}(情况|进展|成效|如何|怎么样)", normalized):
            return _ACTIVITY_TEMPLATE_NAME

    return ""


def is_activity_query(question: str) -> bool:
    """判断是否为“活动报告/普法责任制履职报告”查询。"""
    return bool(resolve_activity_template(question))


def extract_unit_name(question: str) -> str:
    """从问句中提取单位名称，提取失败返回“该单位”。"""
    if not question:
        return "该单位"

    normalized = question.strip()
    suffix_pattern = r"(?:活动开展情况|活动情况|活动开展得怎么样|开展情况|普法责任制履职情况|责任制履职情况|责任制履职|履职情况)"
    query_prefix_pattern = r"(?:查询|请查询|请问|帮我查(?:询)?)?"

    patterns = [
        rf"{query_prefix_pattern}\s*([^，。？?\s]{{1,30}}?单位)的?{suffix_pattern}",
        rf"{query_prefix_pattern}\s*([^，。？?\s]{{1,30}}?单位){suffix_pattern}",
        rf"{query_prefix_pattern}\s*([^，。？?\s]{{1,30}}?)(?:的)?(?:活动|普法责任制|责任制)(?:开展情况|情况|开展得怎么样|履职情况|履职)",
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
