"""
common/base_dataset.py
BaseDataset：所有任务的 Dataset 公共基类
"""
import os
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    提供公共逻辑：
    - 路径验证
    - repeat_factor（虚拟扩充数据集长度）
    - 统一 __len__
    子类需实现 load_samples() 和 __getitem__()
    """

    def __init__(self, data_path: str, sce: str = 'train',
                 transform=None, debug: bool = False,
                 random_state: int = 42, repeat_factor: int = 1, **kwargs):
        if not data_path:
            raise ValueError("data_path cannot be empty")
        abs_path = os.path.abspath(data_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Path not found: {abs_path}")

        if sce not in ['train', 'val', 'test', 'debug']:
            raise ValueError("sce must be one of 'train', 'val', 'test', 'debug'")
        if not isinstance(random_state, int):
            raise TypeError("random_state must be an integer")
        if not isinstance(repeat_factor, int) or repeat_factor < 1:
            raise ValueError("repeat_factor must be a positive integer")

        self.data_path = abs_path
        self.sce = sce
        self.transform = transform
        self.debug = debug
        self.random_state = random_state
        self.repeat_factor = repeat_factor

        self.samples = self.load_samples()
        if debug:
            self.samples = self.samples[:10]

    @abstractmethod
    def load_samples(self) -> list:
        """加载并返回样本列表（路径、标签等），子类实现。"""
        ...

    @abstractmethod
    def __getitem__(self, idx):
        ...

    def __len__(self):
        return len(self.samples) * self.repeat_factor
