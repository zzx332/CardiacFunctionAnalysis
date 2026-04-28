"""tasks/seg2d/__init__.py"""
from tasks import register_task
from .runner import Seg2DRunner
register_task('seg2d')(Seg2DRunner)
