from datetime import datetime
from types import SimpleNamespace

from apps.chat.task.activity_query_extractor import ActivityQueryExtractor
from apps.chat.task.activity_service import ActivityService
from apps.chat.task.question_classifier import (
    extract_unit_name,
    is_activity_query,
    resolve_activity_template,
)


def test_is_activity_query():
    assert is_activity_query("查询XX单位的活动开展情况")
    assert is_activity_query("XX单位活动开展得怎么样")
    assert is_activity_query("XX单位的活动情况")
    assert is_activity_query("市委组织部的活动开展情况")
    assert is_activity_query("请问市委组织部开展普法活动的情况如何")
    assert is_activity_query("请查询XX单位普法责任制履职情况")
    assert is_activity_query("XX单位责任制履职情况怎么样")
    assert not is_activity_query("请帮我统计本月销售额")
    assert not is_activity_query("请问XX单位负责人是谁")


def test_resolve_activity_template():
    assert resolve_activity_template("查询XX单位的活动开展情况") == "activity_report.md"
    assert resolve_activity_template("XX单位活动开展情况如何") == "activity_report.md"
    assert resolve_activity_template("请查询XX单位普法责任制履职情况") == "pufazerenzhi.md"
    assert resolve_activity_template("XX单位责任制履职情况") == "pufazerenzhi.md"
    assert resolve_activity_template("请帮我统计本月销售额") == ""


def test_extract_unit_name():
    assert extract_unit_name("查询XX单位的活动开展情况") == "XX单位"
    assert extract_unit_name("XX单位活动开展得怎么样") == "XX单位"
    assert extract_unit_name("请问YY单位的活动情况") == "YY单位"
    assert extract_unit_name("请查询XX单位普法责任制履职情况") == "XX单位"
    assert extract_unit_name("YY单位责任制履职情况如何") == "YY单位"


def test_render_activity_report():
    service = ActivityService(SimpleNamespace(type="pg"))
    report = service.render_activity_report(
        unit_name="XX单位",
        data={
            "免费法律咨询": "3",
            "现场咨询": "5",
            "法律法规": "宪法",
            "普法宣传活动": "2",
        },
    )

    assert "XX单位法律活动开展情况报告" in report
    assert "免费提供法律咨询 3 人次" in report
    assert "提供现场咨询 5 次" in report


def test_render_pufazerenzhi_report():
    service = ActivityService(SimpleNamespace(type="pg"))
    report = service.render_activity_report(
        unit_name="XX单位",
        data={
            "普法宣传活动": "2",
            "法律法规": "宪法",
        },
        template_name="pufazerenzhi.md",
    )

    assert "XX单位普法责任制履职情况报告" in report
    assert "普法宣传活动" in report
    assert "宪法" in report


def test_render_activity_report_empty_data():
    service = ActivityService(SimpleNamespace(type="pg"))
    report = service.render_activity_report(
        unit_name="XX单位",
        data={},
    )
    assert report == "暂无相关活动开展情况"


def test_render_pufazerenzhi_report_empty_data():
    service = ActivityService(SimpleNamespace(type="pg"))
    report = service.render_activity_report(
        unit_name="XX单位",
        data={},
        template_name="pufazerenzhi.md",
    )
    assert report == "暂无相关普法责任制履职情况"


def test_render_pufazerenzhi_report_no_matching_keywords():
    service = ActivityService(SimpleNamespace(type="pg"))
    report = service.render_activity_report(
        unit_name="XX单位",
        data={
            "免费法律咨询": "3",
            "现场咨询": "5",
        },
        template_name="pufazerenzhi.md",
    )
    assert report == "暂无相关普法责任制履职情况"


def test_query_activity_data_by_keyword(monkeypatch):
    def fake_exec_sql(_self, sql):
        return {
            "fields": ["keyword_answer", "service_logic", "result_value"],
            "data": [
                {"keyword_answer": "免费法律咨询", "service_logic": "0", "result_value": "1"},
                {"keyword_answer": "免费法律咨询", "service_logic": "0", "result_value": "2"},
                {"keyword_answer": "法律法规", "service_logic": "1", "result_value": "宪法"},
                {"keyword_answer": "普法宣传活动", "service_logic": "2", "result_value": "1"},
            ],
        }

    monkeypatch.setattr(ActivityService, "_exec_sql", fake_exec_sql)

    service = ActivityService(SimpleNamespace(type="pg"))
    row = service.query_activity_data("XX单位")

    assert row == {
        "免费法律咨询": "3",
        "法律法规": "宪法",
        "普法宣传活动": "1",
    }


def test_query_activity_data_boolean_count(monkeypatch):
    def fake_exec_sql(_self, sql):
        return {
            "fields": ["keyword_answer", "service_logic", "result_value"],
            "data": [
                {"keyword_answer": "普法宣传活动", "service_logic": "2", "result_value": "1"},
                {"keyword_answer": "普法宣传活动", "service_logic": "2", "result_value": "0"},
                {"keyword_answer": "普法宣传活动", "service_logic": "2", "result_value": "2"},
                {"keyword_answer": "普法宣传活动", "service_logic": "2", "result_value": "0"},
            ],
        }

    monkeypatch.setattr(ActivityService, "_exec_sql", fake_exec_sql)

    service = ActivityService(SimpleNamespace(type="pg"))
    row = service.query_activity_data("XX单位")

    assert row == {
        "普法宣传活动": "2",
    }


def test_query_activity_data_concat_deduplicate_and_law_clean(monkeypatch):
    def fake_exec_sql(_self, sql):
        return {
            "fields": ["keyword_answer", "service_logic", "result_value"],
            "data": [
                {"keyword_answer": "法律法规", "service_logic": "1", "result_value": "《宪法》,《民法典》"},
                {"keyword_answer": "法律法规", "service_logic": "1", "result_value": "民法典，行政处罚法"},
                {"keyword_answer": "媒体报道", "service_logic": "1", "result_value": "法治日报,电视台"},
                {"keyword_answer": "媒体报道", "service_logic": "1", "result_value": "法治日报"},
                {"keyword_answer": "媒体报道", "service_logic": "1", "result_value": "电视台，融媒体中心"},
            ],
        }

    monkeypatch.setattr(ActivityService, "_exec_sql", fake_exec_sql)

    service = ActivityService(SimpleNamespace(type="pg"))
    row = service.query_activity_data("XX单位")

    assert row["法律法规"] == "宪法、民法典、行政处罚法"
    assert row["媒体报道"] == "法治日报、电视台、融媒体中心"


def test_query_activity_data_media_report_deduplicate(monkeypatch):
    def fake_exec_sql(_self, sql):
        return {
            "fields": ["keyword_answer", "service_logic", "result_value"],
            "data": [
                {"keyword_answer": "媒体报道", "service_logic": "1", "result_value": "法治日报,法治日报,电视台"},
                {"keyword_answer": "媒体报道", "service_logic": "1", "result_value": "电视台，融媒体中心"},
                {"keyword_answer": "媒体报道", "service_logic": "1", "result_value": "融媒体中心"},
            ],
        }

    monkeypatch.setattr(ActivityService, "_exec_sql", fake_exec_sql)

    service = ActivityService(SimpleNamespace(type="pg"))
    row = service.query_activity_data("XX单位")

    assert row["媒体报道"] == "法治日报、电视台、融媒体中心"


def test_query_activity_data_with_month_filter(monkeypatch):
    captured = {"sql": ""}

    def fake_exec_sql(_self, sql):
        captured["sql"] = sql
        return {"fields": [], "data": []}

    monkeypatch.setattr(ActivityService, "_exec_sql", fake_exec_sql)

    service = ActivityService(SimpleNamespace(type="pg"))
    service.query_activity_data("XX单位", year=2025, month=3)

    assert "YEAR(publish_time) = 2025" in captured["sql"]
    assert "MONTH(publish_time) = 3" in captured["sql"]


def test_query_activity_data_with_year_filter(monkeypatch):
    captured = {"sql": ""}

    def fake_exec_sql(_self, sql):
        captured["sql"] = sql
        return {"fields": [], "data": []}

    monkeypatch.setattr(ActivityService, "_exec_sql", fake_exec_sql)

    service = ActivityService(SimpleNamespace(type="pg"))
    service.query_activity_data("XX单位", year=2026)

    assert "YEAR(publish_time) = 2026" in captured["sql"]
    assert "MONTH(publish_time)" not in captured["sql"]


def test_activity_query_extractor_extract_success():
    class FakeLLM:
        def __init__(self):
            self.last_messages = []

        def invoke(self, messages):
            self.last_messages = messages
            return SimpleNamespace(content='{"unit_name":"市委组织部","year":2025,"month":3}')

    llm = FakeLLM()
    extractor = ActivityQueryExtractor(llm=llm, lang="简体中文")
    result = extractor.extract(
        question="请查询市委组织部2025-03-07 10:26:04的活动开展情况",
        current_time=datetime(2026, 4, 2, 9, 30, 0),
    )

    assert result.unit_name == "市委组织部"
    assert result.year == 2025
    assert result.month == 3
    assert result.start_year is None
    assert result.end_year is None
    assert any("2026-04-02" in msg.content for msg in llm.last_messages)


def test_activity_query_extractor_default_current_year_when_question_has_no_year():
    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(content='{"unit_name":"市委组织部","month":3}')

    extractor = ActivityQueryExtractor(llm=FakeLLM(), lang="简体中文")
    result = extractor.extract(
        question="请查询市委组织部3月活动开展情况",
        current_time=datetime(2026, 4, 2, 9, 30, 0),
    )

    assert result.unit_name == "市委组织部"
    assert result.year == 2026
    assert result.month == 3


def test_activity_query_extractor_range_fallback_for_recent_three_months():
    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(content='{"unit_name":"市委组织部","year":2026,"month":null}')

    extractor = ActivityQueryExtractor(llm=FakeLLM(), lang="简体中文")
    result = extractor.extract(
        question="请查询市委组织部近三月活动开展情况",
        current_time=datetime(2026, 4, 2, 9, 30, 0),
    )

    assert result.unit_name == "市委组织部"
    assert result.year == 2026
    assert result.month is None
    assert result.start_year is None
    assert result.start_month is None
    assert result.end_year is None
    assert result.end_month is None
    assert result.start_date == "2026-02-02"
    assert result.end_date == "2026-04-02"


def test_activity_query_extractor_range_fallback_for_recent_half_year():
    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(content='{"unit_name":"市委组织部","year":2026,"month":null}')

    extractor = ActivityQueryExtractor(llm=FakeLLM(), lang="简体中文")
    result = extractor.extract(
        question="请查询市委组织部最近半年活动开展情况",
        current_time=datetime(2026, 4, 2, 9, 30, 0),
    )

    assert result.unit_name == "市委组织部"
    assert result.year == 2026
    assert result.month is None
    assert result.start_date == "2025-11-02"
    assert result.end_date == "2026-04-02"


def test_activity_query_extractor_range_fallback_for_month_to_month_without_year():
    class FakeLLM:
        def invoke(self, messages):
            return SimpleNamespace(content='{"unit_name":"市委组织部","year":2026,"month":null}')

    extractor = ActivityQueryExtractor(llm=FakeLLM(), lang="简体中文")
    result = extractor.extract(
        question="请查询市委组织部一月到三月活动开展情况",
        current_time=datetime(2026, 4, 2, 9, 30, 0),
    )

    assert result.unit_name == "市委组织部"
    assert result.year == 2026
    assert result.month is None
    assert result.start_year == 2026
    assert result.start_month == 1
    assert result.end_year == 2026
    assert result.end_month == 3


def test_query_activity_data_with_time_range_filter(monkeypatch):
    captured = {"sql": ""}

    def fake_exec_sql(_self, sql):
        captured["sql"] = sql
        return {"fields": [], "data": []}

    monkeypatch.setattr(ActivityService, "_exec_sql", fake_exec_sql)

    service = ActivityService(SimpleNamespace(type="pg"))
    service.query_activity_data(
        "XX单位",
        year=2026,
        month=3,
        start_date="2026-02-02",
        end_date="2026-04-02",
    )

    assert "publish_time >= '2026-02-02 00:00:00'" in captured["sql"]
    assert "publish_time < '2026-04-03 00:00:00'" in captured["sql"]
    assert "MONTH(publish_time)" not in captured["sql"]
