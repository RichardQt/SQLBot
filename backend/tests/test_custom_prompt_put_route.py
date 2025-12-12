from main import app


def test_custom_prompt_put_route_enabled():
    """回归测试：确保自定义提示词的 PUT 路由已注册，避免 405 Method Not Allowed。"""

    target_path = "/api/v1/system/custom_prompt"
    methods: set[str] = set()

    for route in app.routes:
        if getattr(route, "path", None) == target_path:
            route_methods = getattr(route, "methods", None)
            if route_methods:
                methods.update(route_methods)

    assert "PUT" in methods


def test_custom_prompt_get_and_delete_routes_enabled():
    """回归测试：确保自定义提示词常用 GET/DELETE 路由已注册，避免 404 Not Found。"""

    expected = {
        ("/api/v1/system/custom_prompt/{prompt_type}/page/{current_page}/{page_size}", "GET"),
        ("/api/v1/system/custom_prompt/{id}", "GET"),
        ("/api/v1/system/custom_prompt", "DELETE"),
    }

    actual: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        route_methods = getattr(route, "methods", None) or []
        if not path:
            continue
        for m in route_methods:
            actual.add((path, m))

    missing = expected - actual
    assert not missing, f"缺少路由/方法注册：{missing}"
