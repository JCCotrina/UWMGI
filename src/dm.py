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
        self.num_classes = 4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        patient = self.data.iloc[ix].case
        id_patient = self.data.iloc[ix].imd
        path_image = self.data.iloc[ix].path_img
        path_mask = self.data.iloc[ix].path_mask
        channel = self.data.iloc[ix].channel

        img = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE).astype('float32')[...,channel]
        norm_image = cv2.normalize(img, None, alpha=0, beta=255,norm_type= cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mask = cv2.imread(path_mask, cv2.IMREAD_UNCHANGED).astype(int)

        if self.trans:
            t = self.trans(image=norm_image, mask=mask)
            norm_image = t['image']
            mask = t['mask'] 

        img_t = torch.from_numpy(norm_image).float().unsqueeze(0)
        mask_oh = torch.nn.functional.one_hot(torch.from_numpy(mask).long(), self.num_classes).permute(2,0,1).float()

        return id_patient, img_t, mask_oh


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        path=os.path.join('./data/train'),
        file=os.path.join('./data/data_procesada.csv'),
        val_split=0.7, # 120 / 40 / 40
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
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.val_with_train = val_with_train
        self.train_trans = train_trans
        self.val_trans = val_trans


    def setup(self, stage=None):
        # get list of patients
        data = pd.read_csv(self.file)
        len_data = len(data.index)
        # train / val splits
        train = data.sample(n=int(len_data*val_split), random_state=22)
        val = data.drop(train.index, axis=0)

        train.patient = train.patient.astype(str).str.zfill(3)
        val.patient = val.patient.astype(str).str.zfill(3)

        if self.val_with_train:
            val = train

        # datasets
        self.train_ds = Dataset(
            self.path,
            train,
            trans = A.Compose([
                getattr(A, trans)(**params) for trans, params in self.train_trans.items()
            ]) if self.train_trans else None
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    def __repr__(self):
        aux = pd.read_csv(self.file)
        print(aux.head())
        return str(len(aux))
