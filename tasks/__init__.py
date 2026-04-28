"""
tasks/__init__.py
任务注册表。每个任务子模块通过 @register_task('name') 注册自己。
"""
from typing import Dict, Type

TASK_REGISTRY: Dict[str, Type] = {}


def register_task(name: str):
    """装饰器：将 TaskRunner 类注册到全局注册表。"""
    def decorator(cls):
        TASK_REGISTRY[name] = cls
        return cls
    return decorator


# 导入各任务以触发注册（顺序无关）
from . import landmark  # noqa: F401, E402
from . import view      # noqa: F401, E402
from . import seg2d     # noqa: F401, E402
from . import seg3d     # noqa: F401, E402
from . import strain    # noqa: F401, E402
from . import cardiodx  # noqa: F401, E402
