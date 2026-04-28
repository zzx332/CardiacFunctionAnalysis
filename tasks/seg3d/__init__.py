"""tasks/seg3d/__init__.py"""
from tasks import register_task
from .runner import Seg3DRunner
register_task('seg3d')(Seg3DRunner)
