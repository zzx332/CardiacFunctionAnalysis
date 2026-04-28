"""
tasks/strain/__init__.py
注册 strain（心肌配准/应变）任务
"""
from tasks import register_task
from .runner import StrainRunner
register_task('strain')(StrainRunner)
