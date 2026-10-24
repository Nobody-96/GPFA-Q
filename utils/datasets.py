import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class CacheDataset(Dataset):
    def __init__(self, datas) -> None:
        super().__init__()
        self.datas = datas

    def __getitem__(self, idx):
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)


class ViTImageNetLoaderGenerator:
    """ImageNet loaders that use the timm transforms of the given model."""
    def __init__(self, root, model, val_batch_size, num_workers):
        self.root = root
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self._train_set = None
        self._val_set = None
        config = resolve_data_config(model.default_cfg, model=model)
        self.train_transform = create_transform(**config, is_training=True)
        self.val_transform = create_transform(**config)

    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = ImageFolder(os.path.join(self.root, 'train'), self.train_transform)
        return self._train_set

    @property
    def val_set(self):
        if self._val_set is None:
            self._val_set = ImageFolder(os.path.join(self.root, 'val'), self.val_transform)
        return self._val_set

    def val_loader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False,
                                           num_workers=self.num_workers, pin_memory=False, drop_last=False)

    def calib_loader(self, num, batch_size, seed):
        """`num` training images (training transforms), sampled with `seed` and kept in memory."""
        np.random.seed(seed)
        inds = np.random.permutation(len(self.train_set))[:num]
        calib_set = CacheDataset([self.train_set[i] for i in inds])
        return torch.utils.data.DataLoader(calib_set, batch_size=batch_size, shuffle=False,
                                           num_workers=self.num_workers, pin_memory=True, drop_last=False)
