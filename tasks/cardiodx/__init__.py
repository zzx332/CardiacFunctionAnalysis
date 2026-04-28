"""
tasks/cardiodx/__init__.py
注册 cardiodx（心脏疾病诊断二分类）任务
"""
from tasks import register_task
from .runner import CardioDxRunner
register_task('cardiodx')(CardioDxRunner)
