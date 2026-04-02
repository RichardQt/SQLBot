from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import calendar
import re
from typing import Optional

import orjson
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from apps.template.activity_extract.generator import get_activity_extract_template
from common.utils.utils import SQLBotLogUtil, extract_nested_json


@dataclass
class ActivityQueryExtractResult:
    """活动问题提取结果。"""

    unit_name: str
    year: int
    month: Optional[int]
    start_year: Optional[int] = None
    start_month: Optional[int] = None
    end_year: Optional[int] = None
    end_month: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ActivityQueryExtractor:
    """通过大模型提取活动查询参数（单位名称、年份、月份、时间范围）。"""

    _MONTH_WORD_MAP = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "十一": 11,
        "十二": 12,
    }

    _NUMBER_WORD_MAP = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "十一": 11,
        "十二": 12,
        "十三": 13,
        "十四": 14,
        "十五": 15,
        "十六": 16,
        "十七": 17,
        "十八": 18,
        "十九": 19,
        "二十": 20,
        "二十一": 21,
        "二十二": 22,
        "二十三": 23,
        "二十四": 24,
    }

    def __init__(self, llm: BaseChatModel, lang: str):
        self.llm = llm
        self.lang = lang

    @staticmethod
    def _normalize_month(raw_month: object) -> Optional[int]:
        if raw_month is None:
            return None

        if isinstance(raw_month, bool):
            return None

        try:
            month = int(str(raw_month).strip())
        except (ValueError, TypeError):
            return None

        if 1 <= month <= 12:
            return month
        return None

    @staticmethod
    def _normalize_year(raw_year: object) -> Optional[int]:
        if raw_year is None:
            return None

        if isinstance(raw_year, bool):
            return None

        try:
            year = int(str(raw_year).strip())
        except (ValueError, TypeError):
            return None

        if 1900 <= year <= 2100:
            return year
        return None

    @classmethod
    def _normalize_month_token(cls, raw_token: str) -> Optional[int]:
        token = str(raw_token or "").strip().replace("月", "")
        if not token:
            return None

        if token.isdigit():
            month = int(token)
            return month if 1 <= month <= 12 else None

        if token in cls._MONTH_WORD_MAP:
            return cls._MONTH_WORD_MAP[token]

        return None

    @classmethod
    def _normalize_count_token(cls, raw_token: str) -> Optional[int]:
        token = str(raw_token or "").strip()
        if not token:
            return None

        if token.isdigit():
            value = int(token)
            return value if value > 0 else None

        return cls._NUMBER_WORD_MAP.get(token)

    @staticmethod
    def _shift_year_month(year: int, month: int, delta_months: int) -> tuple[int, int]:
        month_index = year * 12 + (month - 1) + delta_months
        shifted_year = month_index // 12
        shifted_month = month_index % 12 + 1
        return shifted_year, shifted_month

    @staticmethod
    def _shift_date_by_months(base_date: date, delta_months: int) -> date:
        month_index = base_date.year * 12 + (base_date.month - 1) + delta_months
        target_year = month_index // 12
        target_month = month_index % 12 + 1
        max_day = calendar.monthrange(target_year, target_month)[1]
        target_day = min(base_date.day, max_day)
        return date(target_year, target_month, target_day)

    @staticmethod
    def _normalize_date_string(raw_date: object) -> Optional[str]:
        if raw_date is None:
            return None
        text = str(raw_date).strip()
        if not text:
            return None
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d").date()
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            return None

    @staticmethod
    def _is_valid_range(
        start_year: Optional[int],
        start_month: Optional[int],
        end_year: Optional[int],
        end_month: Optional[int],
    ) -> bool:
        if None in (start_year, start_month, end_year, end_month):
            return False
        if not (1900 <= int(start_year) <= 2100 and 1900 <= int(end_year) <= 2100):
            return False
        if not (1 <= int(start_month) <= 12 and 1 <= int(end_month) <= 12):
            return False
        return (int(start_year), int(start_month)) <= (int(end_year), int(end_month))

    @classmethod
    def _extract_range_by_fallback(
        cls,
        question: str,
        default_year: int,
        now: datetime,
    ) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[str], Optional[str]]:
        normalized = re.sub(r"\s+", "", question or "")

        # 1) 近N月 / 最近N个月
        if re.search(r"(?:近|最近)半年", normalized):
            end_date_obj = now.date()
            start_date_obj = cls._shift_date_by_months(end_date_obj, -(6 - 1))
            return (
                None,
                None,
                None,
                None,
                start_date_obj.strftime("%Y-%m-%d"),
                end_date_obj.strftime("%Y-%m-%d"),
            )

        recent_match = re.search(r"(?:近|最近)([0-9一二三四五六七八九十]{1,3})个?月", normalized)
        if recent_match:
            month_count = cls._normalize_count_token(recent_match.group(1))
            if month_count and month_count >= 1:
                end_date_obj = now.date()
                start_date_obj = cls._shift_date_by_months(end_date_obj, -(month_count - 1))
                return (
                    None,
                    None,
                    None,
                    None,
                    start_date_obj.strftime("%Y-%m-%d"),
                    end_date_obj.strftime("%Y-%m-%d"),
                )

        # 2) YYYY年M月到YYYY年N月
        full_year_range_match = re.search(
            r"((?:19|20)\d{2})年([0-9一二三四五六七八九十]{1,3})月?(?:到|至|-|~|～)((?:19|20)\d{2})年([0-9一二三四五六七八九十]{1,3})月?",
            normalized,
        )
        if full_year_range_match:
            sy = cls._normalize_year(full_year_range_match.group(1))
            sm = cls._normalize_month_token(full_year_range_match.group(2))
            ey = cls._normalize_year(full_year_range_match.group(3))
            em = cls._normalize_month_token(full_year_range_match.group(4))
            if cls._is_valid_range(sy, sm, ey, em):
                return sy, sm, ey, em, None, None

        # 3) M月到N月（可带单个年份）
        explicit_year_match = re.search(r"((?:19|20)\d{2})年", normalized)
        target_year = cls._normalize_year(explicit_year_match.group(1)) if explicit_year_match else default_year

        month_range_match = re.search(
            r"([0-9一二三四五六七八九十]{1,3})月?(?:到|至|-|~|～)([0-9一二三四五六七八九十]{1,3})月",
            normalized,
        )
        if month_range_match:
            sm = cls._normalize_month_token(month_range_match.group(1))
            em = cls._normalize_month_token(month_range_match.group(2))
            if sm and em:
                sy = target_year
                ey = target_year if sm <= em else target_year + 1
                if cls._is_valid_range(sy, sm, ey, em):
                    return sy, sm, ey, em, None, None

        return None, None, None, None, None, None

    def extract(self, question: str, current_time: Optional[datetime] = None) -> ActivityQueryExtractResult:
        now = current_time or datetime.now()
        template = get_activity_extract_template()

        format_kwargs = {
            "question": question,
            "lang": self.lang,
            "current_date": now.strftime("%Y-%m-%d"),
            "current_month_day": now.strftime("%m-%d"),
            "current_year": now.year,
        }

        system_prompt_tpl = template.get("system")
        if not system_prompt_tpl:
            raise ValueError("activity_extract.system 模板缺失")

        messages = [SystemMessage(content=system_prompt_tpl.format(**format_kwargs))]

        user_prompt_tpl = template.get("user")
        if user_prompt_tpl:
            messages.append(HumanMessage(content=user_prompt_tpl.format(**format_kwargs)))

        default_result = ActivityQueryExtractResult(unit_name="该单位", year=now.year, month=None)

        try:
            response = self.llm.invoke(messages)
            content = response.content
            SQLBotLogUtil.info(f"[activity] 参数提取模型原始响应: {content}")

            json_str = extract_nested_json(content)
            if not json_str:
                return default_result

            data = orjson.loads(json_str)
            unit_name = str(data.get("unit_name", "") or "").strip() or "该单位"
            year = self._normalize_year(data.get("year")) or now.year
            month = self._normalize_month(data.get("month"))

            start_year = self._normalize_year(data.get("start_year"))
            start_month = self._normalize_month(data.get("start_month"))
            end_year = self._normalize_year(data.get("end_year"))
            end_month = self._normalize_month(data.get("end_month"))
            start_date = self._normalize_date_string(data.get("start_date"))
            end_date = self._normalize_date_string(data.get("end_date"))

            # 模型仅给出月份范围时，补齐年份（默认当前/提取年份）
            if start_month and end_month:
                if start_year is None:
                    start_year = year
                if end_year is None:
                    end_year = start_year if start_month <= end_month else start_year + 1

            has_valid_date_range = bool(start_date and end_date and start_date <= end_date)

            # 若模型没提取到范围，走规则兜底识别（近三月、一月到三月）
            if not has_valid_date_range and not self._is_valid_range(start_year, start_month, end_year, end_month):
                (
                    start_year,
                    start_month,
                    end_year,
                    end_month,
                    start_date,
                    end_date,
                ) = self._extract_range_by_fallback(
                    question=question,
                    default_year=year,
                    now=now,
                )
                has_valid_date_range = bool(start_date and end_date and start_date <= end_date)

            if has_valid_date_range:
                month = None
                start_year = None
                start_month = None
                end_year = None
                end_month = None
            elif self._is_valid_range(start_year, start_month, end_year, end_month):
                month = None
                start_date = None
                end_date = None
            else:
                start_year = None
                start_month = None
                end_year = None
                end_month = None
                start_date = None
                end_date = None

            result = ActivityQueryExtractResult(
                unit_name=unit_name,
                year=year,
                month=month,
                start_year=start_year,
                start_month=start_month,
                end_year=end_year,
                end_month=end_month,
                start_date=start_date,
                end_date=end_date,
            )
            SQLBotLogUtil.info(
                "[activity] 参数提取解析完成: "
                f"unit_name={result.unit_name}, year={result.year}, month={result.month}, "
                f"start={result.start_year}-{result.start_month}, end={result.end_year}-{result.end_month}, "
                f"start_date={result.start_date}, end_date={result.end_date}"
            )
            return result
        except Exception as exc:
            SQLBotLogUtil.warning(f"[activity] 参数提取失败，使用兜底值: err={exc}")
            return default_result
