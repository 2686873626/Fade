import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision import transforms as T
from typing import Optional

from .ffpp_dataset_v2 import FF_Dataset
from .celeb_dataset import Celeb_Dataset

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class FFData(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self.train_set = FF_Dataset(
                list_path=os.path.join(self.opt.list_path, 'train.txt'),
                data_path=os.path.join(self.opt.data_path, self.opt.img_quality),
                num_samples=self.opt.num_samples,
                interval=self.opt.interval,
                img_size=self.opt.img_size,
                random_sample=self.opt.random_sample
            )
            self.val_set = FF_Dataset(
                list_path=os.path.join(self.opt.list_path, 'test_all.txt'),
                data_path=os.path.join(self.opt.data_path, self.opt.img_quality),
                num_samples=self.opt.num_samples,
                interval=self.opt.interval,
                img_size=self.opt.img_size,
            )
            
        if stage in (None, 'test'):
            # self.test_set = FF_Dataset(
            #     list_path=os.path.join(self.opt.list_path, 'test_all.txt'),
            #     data_path=os.path.join(self.opt.data_path, self.opt.img_quality),
            #     num_samples=self.opt.num_samples,
            #     interval=self.opt.interval,
            #     img_size=self.opt.img_size,
            # )

            self.test_set = Celeb_Dataset(
                version='v2',
                num_samples=8,
                interval=3,
                img_size=self.opt.img_size
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.opt.train_batch_size, shuffle=True,
            num_workers=self.opt.num_workers, pin_memory=False,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.opt.val_batch_size,
            num_workers=self.opt.num_workers, pin_memory=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.opt.val_batch_size,
            num_workers=self.opt.num_workers, pin_memory=False,
        )
