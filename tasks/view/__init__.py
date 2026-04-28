"""tasks/view/__init__.py"""
from tasks import register_task
from .runner import ViewRunner
register_task('view')(ViewRunner)
