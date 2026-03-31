from types import SimpleNamespace

from apps.chat.task.activity_service import ActivityService
from apps.chat.task.question_classifier import extract_unit_name, is_activity_query


def test_is_activity_query():
    assert is_activity_query("查询XX单位的活动开展情况")
    assert is_activity_query("XX单位活动开展得怎么样")
    assert is_activity_query("XX单位的活动情况")
    assert is_activity_query("市委组织部的活动开展情况")
    assert is_activity_query("请问市委组织部开展普法活动的情况如何")
    assert not is_activity_query("请帮我统计本月销售额")


def test_extract_unit_name():
    assert extract_unit_name("查询XX单位的活动开展情况") == "XX单位"
    assert extract_unit_name("XX单位活动开展得怎么样") == "XX单位"
    assert extract_unit_name("请问YY单位的活动情况") == "YY单位"


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

    assert "XX单位普法与法律活动开展情况报告" in report
    assert "免费提供法律咨询3 人次" in report
    assert "法律法规" in report
    assert "宪法" in report


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
