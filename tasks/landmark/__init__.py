"""
tasks/landmark/__init__.py
注册 landmark 任务。
"""
from tasks import register_task
from .runner import LandmarkRunner

register_task('landmark')(LandmarkRunner)
