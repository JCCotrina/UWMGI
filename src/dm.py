import torch
import nibabel as nib
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations as A
import numpy as np
import pandas as pd
import cv2
import os
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, data, trans=None):
        self.path = path
        self.data = data
        self.trans = trans
        self.path_img = data['path_img'].tolist()
        self.path_mask = data['path_mask'].tolist()
        self.num_classes = 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        path_image = self.path_img[ix]
        path_mask = self.path_mask[ix]

        img = cv2.imread(path_image, cv2.IMREAD_UNCHANGED)
        img = np.tile(img[..., None], [1, 1, 3]).astype('float32')
        norm_image = cv2.normalize(img, None, alpha=0, beta=1,norm_type= cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        mask = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED).astype('float32')/255.0

        if self.trans:
            t = self.trans(image=norm_image, mask=mask)
            norm_image = t['image']
            mask = t['mask'] 

        # img_t = torch.from_numpy(norm_image).permute(2,0,1).float()
        # mask_oh = torch.from_numpy(mask).permute(2,0,1).float()
        img_t = np.transpose(norm_image, (2,0,1))
        mask_oh = np.transpose(mask ,(2,0,1))

        return torch.tensor(img_t), torch.tensor(mask_oh)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        path=os.path.join('../input/uw-madison-gi-tract-image-segmentation/train'),
        file=os.path.join('../input/preprocessing/data_procesada.csv'),
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        shuffle_train=True,
        val_with_train=False,
        train_trans=None,
        val_trans=None,
        **kwargs
    ):
        super().__init__()
        self.path = path
        self.file = file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.val_with_train = val_with_train
        self.train_trans = train_trans
        self.val_trans = val_trans


    def setup(self,fold=0, stage=None):
        # get list of patients
        data = pd.read_csv(self.file)

        train = data.query("fold!=@fold").reset_index(drop=True)
        val = data.query("fold==@fold").reset_index(drop=True)

        if self.val_with_train:
            val = train

        # datasets
        self.train_ds = Dataset(
            self.path,
            train,
            # trans = A.Compose([
            #     getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            # ]) if self.train_trans else None
            trans =A.Compose([
                    A.Resize( height=224, width=224, interpolation=cv2.INTER_NEAREST),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
                    A.OneOf([
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            # #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                        A.GaussianBlur(),
                    ], p=0.25),
                    A.CoarseDropout(max_holes=8, max_height=11, max_width=11,
                                    min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
                    ], p=1.0),
        )

        self.val_ds = Dataset(
            self.path,
            val,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.val_trans.items()
            ]) if self.val_trans else None
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_train,
            pin_memory=self.pin_memory,
            drop_last=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=2*self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )