from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Any, Optional

from apps.datasource.models.datasource import CoreDatasource
from apps.system.schemas.system_schema import AssistantOutDsSchema
from common.utils.utils import SQLBotLogUtil


class ActivityService:
    """活动开展情况查询与报告渲染服务。"""

    _DEFAULT_TEMPLATE = "activity_report.md"
    _ALLOWED_TEMPLATES = {"activity_report.md", "pufazerenzhi.md"}
    _EMPTY_REPORT_MESSAGE = {
        "activity_report.md": "暂无相关活动开展情况",
        "pufazerenzhi.md": "暂无相关普法责任制履职情况",
    }

    def _get_empty_report_message(self, template_name: str) -> str:
        return self._EMPTY_REPORT_MESSAGE.get(template_name, "暂无相关活动开展情况")

    def __init__(self, datasource: CoreDatasource | AssistantOutDsSchema):
        self.datasource = datasource
        ds_id = getattr(datasource, "id", None)
        ds_name = getattr(datasource, "name", None)
        ds_type = getattr(datasource, "type", None)
        SQLBotLogUtil.info(
            f"[activity] ActivityService 初始化: ds_id={ds_id}, ds_name={ds_name}, ds_type={ds_type}"
        )

    @staticmethod
    def _escape_sql_literal(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _is_valid_year_month(year: Optional[int], month: Optional[int]) -> bool:
        return isinstance(year, int) and isinstance(month, int) and 1900 <= year <= 2100 and 1 <= month <= 12

    @staticmethod
    def _next_month_start(year: int, month: int) -> tuple[int, int]:
        if month == 12:
            return year + 1, 1
        return year, month + 1

    @staticmethod
    def _normalize_date_string(raw_date: Optional[str]) -> Optional[str]:
        if raw_date is None:
            return None
        text = str(raw_date).strip()
        if not text:
            return None
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d")
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            return None

    def _exec_sql(self, sql: str) -> dict[str, Any]:
        from apps.db.db import exec_sql
        SQLBotLogUtil.info(f"[activity] 即将执行SQL: {sql}")
        result = exec_sql(ds=self.datasource, sql=sql, origin_column=False)
        data_count = 0
        if isinstance(result, dict):
            data = result.get("data")
            data_count = len(data) if isinstance(data, list) else 0
        SQLBotLogUtil.info(f"[activity] SQL执行完成: 返回行数={data_count}")
        return result

    @staticmethod
    def _strip_logic_hints(report: str) -> str:
        """移除模板中的逻辑提示说明。"""
        return re.sub(r"（逻辑[^）]*）", "", report)

    @staticmethod
    def _cleanup_rendered_line(line: str) -> str:
        """清理替换后行内因空值产生的多余标点和空白。"""
        line = re.sub(r"（\s*）", "", line)
        line = re.sub(r"\s{2,}", " ", line)
        line = re.sub(r"、\s*、", "、", line)
        line = re.sub(r"，\s*，", "，", line)
        line = re.sub(r"；\s*；", "；", line)
        line = re.sub(r"：\s*、", "：", line)
        line = re.sub(r"、\s*。", "。", line)
        line = re.sub(r"，\s*。", "。", line)
        line = re.sub(r"；\s*。", "。", line)
        return line.strip()

    def _render_template_by_keyword_lines(self, template: str, data: dict[str, str]) -> str:
        """按行渲染模板：仅保留命中关键词的占位符行，避免整段误删。"""
        rendered_lines: list[str] = []
        for line in template.splitlines():
            tokens = re.findall(r"\[([^\]]+)\]", line)
            if not tokens:
                rendered_lines.append(line)
                continue

            hit_count = 0
            new_line = line
            for token_name in tokens:
                token = f"[{token_name}]"
                value = str(data.get(token_name, "") or "").strip()
                if value:
                    hit_count += 1
                    new_line = new_line.replace(token, value)
                else:
                    new_line = new_line.replace(token, "")

            # 本行没有任何关键词命中：整行丢弃
            if hit_count == 0:
                continue

            new_line = self._cleanup_rendered_line(new_line)
            if new_line:
                rendered_lines.append(new_line)

        return "\n".join(rendered_lines)

    @staticmethod
    def _remove_placeholder_lines(report: str) -> str:
        """移除仍包含占位符的行（表示该部分无数据，不应展示）。"""
        lines = report.splitlines()
        before_count = len(lines)
        kept: list[str] = []
        for line in lines:
            if re.search(r"\[[^\]]+\]", line):
                continue
            kept.append(line)

        cleaned = "\n".join(kept)

        # 清理空的小节标题（例如：#### xxx 后无内容）
        # 注意：不能使用 DOTALL 贪婪正则，否则会误删整段正文。
        section_lines = cleaned.splitlines()
        compact_lines: list[str] = []
        idx = 0
        while idx < len(section_lines):
            line = section_lines[idx]
            if line.startswith("#### "):
                probe = idx + 1
                has_body = False
                while probe < len(section_lines):
                    probe_line = section_lines[probe]
                    probe_strip = probe_line.strip()

                    # 到达下一个同级/上级标题或分隔线，停止探测
                    if probe_line.startswith("#### ") or probe_line.startswith("### ") or probe_strip == "---":
                        break

                    if probe_strip:
                        has_body = True
                        break
                    probe += 1

                if not has_body:
                    SQLBotLogUtil.info(f"[activity] 删除空小节标题: {line}")
                    idx += 1
                    continue

            compact_lines.append(line)
            idx += 1

        cleaned = "\n".join(compact_lines)

        # 清理多余空行
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        after_count = len(cleaned.splitlines()) if cleaned else 0
        SQLBotLogUtil.info(
            f"[activity] 占位符行清理完成: 原始行数={before_count}, 清理后行数={after_count}"
        )
        return cleaned

    @staticmethod
    def _expand_value(value: str, count: int) -> list[str]:
        """将聚合值扩展为给定数量的占位值。"""
        if count <= 0:
            return []
        return [value for _ in range(count)]

    @staticmethod
    def _is_non_zero_value(value: Any) -> bool:
        """判断值是否属于“非0”，用于布尔型计次。"""
        if value is None:
            return False

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return value != 0

        text = str(value).strip()
        if not text:
            return False

        try:
            return float(text) != 0
        except ValueError:
            # 兼容历史脏数据：无法转数值时，非空且不等于"0"也视作一次有效计数
            return text != "0"

    @staticmethod
    def _deduplicate_keep_order(values: list[str]) -> list[str]:
        """对字符串列表去重并保持原有顺序。"""
        seen: set[str] = set()
        deduplicated: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduplicated.append(value)
        return deduplicated

    @staticmethod
    def _normalize_concat_items(keyword: str, value: str) -> list[str]:
        """标准化字符串拼接项。"""
        text = str(value or "").strip()
        if not text:
            return []

        # “法律法规”字段特殊处理：去书名号并统一分隔符为顿号
        if keyword == "法律法规":
            text = text.replace("《", "").replace("》", "")
            text = text.replace(",", "、").replace("，", "、")
            return [item.strip() for item in text.split("、") if item.strip()]

        # “媒体报道”字段特殊处理：统一分隔符为顿号
        if keyword == "媒体报道":
            text = text.replace(",", "、").replace("，", "、")
            return [item.strip() for item in text.split("、") if item.strip()]

        return [text]

    def _aggregate_by_logic(
        self,
        service_logic: int,
        records: list[dict[str, Any]],
        keyword: str = "",
    ) -> str:
        """根据 service_logic 对 result_value 进行聚合。
        
        Args:
            service_logic: 聚合逻辑（0=数值求和，1=字符串拼接，2=布尔判断）
            records: 记录列表
            keyword: 关键词（用于字段特化处理）
            
        Returns:
            聚合后的字符串结果
        """
        if not records:
            SQLBotLogUtil.info(f"[activity] 聚合跳过: logic={service_logic}, records=0")
            return ""

        if service_logic == 0:
            # 逻辑 0：数值相加
            try:
                total = sum(int(r.get("result_value", 0) or 0) for r in records)
                result = str(total) if total else ""
                SQLBotLogUtil.info(
                    f"[activity] 聚合完成: logic=0, records={len(records)}, total={result or 'EMPTY'}"
                )
                return result
            except (ValueError, TypeError):
                SQLBotLogUtil.warning(
                    f"[activity] 聚合失败: logic=0 存在非整数 result_value, records={records}"
                )
                return ""

        elif service_logic == 1:
            # 逻辑 1：字符串拼接
            values: list[str] = []
            for record in records:
                raw_value = str(record.get("result_value", "") or "")
                values.extend(self._normalize_concat_items(keyword=keyword, value=raw_value))

            unique_values = self._deduplicate_keep_order(values)
            result = "、".join(unique_values) if unique_values else ""
            SQLBotLogUtil.info(
                f"[activity] 聚合完成: logic=1, keyword={keyword}, records={len(records)}, kept_values={len(values)}, deduplicated_values={len(unique_values)}, result_len={len(result)}"
            )
            return result

        elif service_logic == 2:
            # 逻辑 2：布尔计次（每条记录 result_value 非0 则计 1 次）
            count = sum(
                1 for r in records if self._is_non_zero_value(r.get("result_value"))
            )
            result = str(count)
            SQLBotLogUtil.info(
                f"[activity] 聚合完成: logic=2, records={len(records)}, non_zero_count={count}"
            )
            return result

        SQLBotLogUtil.warning(f"[activity] 未知聚合逻辑: logic={service_logic}")
        return ""

    def query_activity_data(
        self,
        unit_name: str,
        year: Optional[int] = None,
        month: Optional[int] = None,
        start_year: Optional[int] = None,
        start_month: Optional[int] = None,
        end_year: Optional[int] = None,
        end_month: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, str]:
        """基于系统已选数据源查询活动表，并按 keyword_answer 精确聚合。

        Returns:
            关键词到聚合值的映射，例如：
            {
                "免费法律咨询": "3",
                "法律法规": "宪法",
                "普法宣传活动": "是"
            }
        """
        escaped_unit = self._escape_sql_literal(unit_name)
        like_value = f"%{escaped_unit}%"
        SQLBotLogUtil.info(
            "[activity] 开始查询活动数据: "
            f"unit_name={unit_name}, like={like_value}, year={year}, month={month}, "
            f"start={start_year}-{start_month}, end={end_year}-{end_month}, "
            f"start_date={start_date}, end_date={end_date}"
        )

        # 查询匹配的所有记录
        sql = (
            "select keyword_answer, service_logic, result_value "
            "from fx_education_articles_keyword "
            f"where unit_name like '{like_value}' "
        )

        normalized_start_date = self._normalize_date_string(start_date)
        normalized_end_date = self._normalize_date_string(end_date)
        has_valid_date_range = bool(
            normalized_start_date and normalized_end_date and normalized_start_date <= normalized_end_date
        )

        has_valid_range = (
            self._is_valid_year_month(start_year, start_month)
            and self._is_valid_year_month(end_year, end_month)
            and (int(start_year), int(start_month)) <= (int(end_year), int(end_month))
        )

        if has_valid_date_range:
            start_dt = f"{normalized_start_date} 00:00:00"
            end_dt = datetime.strptime(normalized_end_date, "%Y-%m-%d") + timedelta(days=1)
            end_exclusive = end_dt.strftime("%Y-%m-%d 00:00:00")
            sql += (
                f"and publish_time >= '{start_dt}' "
                f"and publish_time < '{end_exclusive}' "
            )
            SQLBotLogUtil.info(
                f"[activity] 已追加按日范围筛选: start={start_dt}, end_exclusive={end_exclusive}"
            )
        elif has_valid_range:
            range_start = f"{int(start_year):04d}-{int(start_month):02d}-01 00:00:00"
            next_year, next_month = self._next_month_start(int(end_year), int(end_month))
            range_end_exclusive = f"{next_year:04d}-{next_month:02d}-01 00:00:00"
            sql += (
                f"and publish_time >= '{range_start}' "
                f"and publish_time < '{range_end_exclusive}' "
            )
            SQLBotLogUtil.info(
                f"[activity] 已追加时间范围筛选: start={range_start}, end_exclusive={range_end_exclusive}"
            )
        else:
            normalized_year: Optional[int] = None
            if isinstance(year, int) and 1900 <= year <= 2100:
                normalized_year = year
                sql += f"and YEAR(publish_time) = {normalized_year} "
                SQLBotLogUtil.info(f"[activity] 已追加年份筛选: year={normalized_year}")
            elif year is not None:
                SQLBotLogUtil.warning(f"[activity] 年份参数无效，忽略年份筛选: year={year}")

            normalized_month: Optional[int] = None
            if isinstance(month, int) and 1 <= month <= 12:
                normalized_month = month
                sql += f"and MONTH(publish_time) = {normalized_month} "
                SQLBotLogUtil.info(f"[activity] 已追加月份筛选: month={normalized_month}")
            elif month is not None:
                SQLBotLogUtil.warning(f"[activity] 月份参数无效，忽略月份筛选: month={month}")

        result = self._exec_sql(sql)
        data = result.get("data") if isinstance(result, dict) else []
        SQLBotLogUtil.info(
            f"[activity] 原始数据读取完成: unit_name={unit_name}, rows={len(data) if isinstance(data, list) else 0}"
        )

        # 按 keyword_answer 分组，同时保留其 service_logic
        keyword_grouped: dict[str, dict[str, Any]] = {}
        for record in data:
            try:
                keyword = str(record.get("keyword_answer", "") or "").strip()
                if not keyword:
                    SQLBotLogUtil.warning(f"[activity] 发现空关键词记录, record={record}")
                    continue

                logic = int(record.get("service_logic", 0) or 0)
                if keyword not in keyword_grouped:
                    keyword_grouped[keyword] = {
                        "logic": logic,
                        "records": [],
                    }

                # 同一关键词出现不同 logic 时，沿用首条并记录告警
                if keyword_grouped[keyword]["logic"] != logic:
                    SQLBotLogUtil.warning(
                        f"[activity] 关键词 logic 不一致, keyword={keyword}, keep={keyword_grouped[keyword]['logic']}, got={logic}"
                    )

                keyword_grouped[keyword]["records"].append(record)
            except (ValueError, TypeError):
                SQLBotLogUtil.warning(f"[activity] 发现非法记录, record={record}")
                pass

        SQLBotLogUtil.info(f"[activity] 关键词分组完成: keyword_count={len(keyword_grouped)}")

        # 按关键词逐个聚合
        keyword_values: dict[str, str] = {}
        for keyword, item in keyword_grouped.items():
            logic = int(item["logic"])
            records = item["records"]
            agg_result = self._aggregate_by_logic(logic, records, keyword=keyword)
            if agg_result:
                keyword_values[keyword] = agg_result
                SQLBotLogUtil.info(
                    f"[activity] 关键词聚合完成: keyword={keyword}, logic={logic}, value={agg_result}"
                )
            else:
                SQLBotLogUtil.info(
                    f"[activity] 关键词聚合为空: keyword={keyword}, logic={logic}"
                )

        SQLBotLogUtil.info(f"[activity] 关键词聚合总结果: {keyword_values}")

        return keyword_values

    def render_activity_report(
        self,
        unit_name: str,
        data: Optional[dict[str, str]] = None,
        template_name: str = _DEFAULT_TEMPLATE,
    ) -> str:
        """使用文本模板渲染活动报告。
        
        Args:
            unit_name: 单位名称
            data: 关键词聚合数据（可选，若不提供则自动查询）
            template_name: 模板文件名，支持 activity_report.md / pufazerenzhi.md
            
        Returns:
            格式化的活动报告字符串
        """
        selected_template = (
            template_name if template_name in self._ALLOWED_TEMPLATES else self._DEFAULT_TEMPLATE
        )
        if selected_template != template_name:
            SQLBotLogUtil.warning(
                f"[activity] 非法模板名，已回退默认模板: requested={template_name}, fallback={selected_template}"
            )

        template_path = Path(__file__).resolve().parents[3] / "templates" / selected_template
        template_content = template_path.read_text(encoding="utf-8")
        SQLBotLogUtil.info(
            f"[activity] 开始渲染报告: template={template_path}, template_len={len(template_content)}"
        )

        # 如果未提供数据，自动查询
        if data is None:
            data = self.query_activity_data(unit_name)

        SQLBotLogUtil.info(f"[activity] 渲染入参: unit_name={unit_name}, data={data}")

        if not data:
            empty_message = self._get_empty_report_message(selected_template)
            SQLBotLogUtil.info(
                f"[activity] 渲染终止: 无可用数据, template={selected_template}, message={empty_message}"
            )
            return empty_message

        report = template_content

        # 标题单位名替换（兼容“某某单位”写法与显式占位符写法）
        report = report.replace("某某单位", unit_name)
        report = report.replace("{unit_name}", unit_name)
        report = report.replace("{generate_time}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        SQLBotLogUtil.info(f"[activity] 关键词数据量: {len(data)}")

        # 先记录命中情况，便于排查“为什么没显示”
        matched_keyword_count = 0
        for keyword, value in data.items():
            token = f"[{keyword}]"
            token_count = report.count(token)
            if token_count > 0:
                matched_keyword_count += 1
                SQLBotLogUtil.info(
                    f"[activity] 关键词命中模板: token={token}, count={token_count}, value={value}"
                )
            else:
                SQLBotLogUtil.info(f"[activity] 关键词未命中模板: token={token}")

        if matched_keyword_count == 0:
            empty_message = self._get_empty_report_message(selected_template)
            SQLBotLogUtil.info(
                f"[activity] 渲染终止: 数据与模板无匹配关键词, template={selected_template}, message={empty_message}"
            )
            return empty_message

        # 按行渲染：无命中行删除，命中行替换
        report = self._render_template_by_keyword_lines(report, data)

        # 去除模板中的逻辑判断提示文本
        report = self._strip_logic_hints(report)

        # 若某部分无数据，仍会保留占位符，统一删除对应行
        report = self._remove_placeholder_lines(report)

        SQLBotLogUtil.info(
            f"[activity] 报告渲染完成: report_len={len(report)}, is_empty={not bool(report.strip())}"
        )
        SQLBotLogUtil.info(f"[activity] 报告预览(前300): {report[:300]}")

        if not report.strip():
            empty_message = self._get_empty_report_message(selected_template)
            SQLBotLogUtil.info(
                f"[activity] 渲染终止: 报告正文为空, template={selected_template}, message={empty_message}"
            )
            return empty_message

        return report
