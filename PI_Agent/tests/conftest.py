"""pytest 配置：tools.py 依赖 asyncio.to_thread，仅用 asyncio 后端"""


def pytest_collection_modifyitems(config, items):
    """过滤掉 trio 后端的测试变体"""
    for item in items[:]:
        if hasattr(item, "callspec"):
            backend = item.callspec.params.get("anyio_backend")
            if backend == "trio":
                items.remove(item)
