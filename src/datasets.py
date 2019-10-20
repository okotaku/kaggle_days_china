import os
import cv2
import torch
import random
import pydicom
import numpy as np
from albumentations import *
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from logger import LOGGER


class KDDataset(Dataset):

    def __init__(self,
                 df,
                 y,
                 img_size,
                 image_path,
                 crop_rate=1.0,
                 id_colname="id",
                 img_type=".png",
                 transforms=None,
                 means=[0.485, 0.456, 0.406],
                 stds=[0.229, 0.224, 0.225],
                 ):
        self.df = df
        self.y = y
        self.img_size = img_size
        self.image_path = image_path
        self.transforms = transforms
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.id_colname = id_colname
        self.img_type = img_type
        self.crop_rate = crop_rate

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        target = self.y[idx]

        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        if target[0] == 0:
            dirs = "nonstar"
        else:
            dirs = "star"
        img_path = os.path.join(self.image_path, dirs, img_id + self.img_type)
        img = cv2.imread(img_path)

        if self.crop_rate < 1:
            img = random_cropping(img, is_random=True, ratio=self.crop_rate)

        if self.transforms is not None:
            augmented = self.transforms(image=img)
            img = augmented['image']

        img = img / 255
        img -= self.means
        img /= self.stds
        img = img.transpose((2, 0, 1))

        return torch.FloatTensor(img), torch.FloatTensor(target)


class KDDatasetTest(Dataset):

    def __init__(self,
                 df,
                 img_size,
                 image_path,
                 crop_rate = 1.0,
                 id_colname="id",
                 img_type=".png",
                 transforms=None,
                 means=[0.485, 0.456, 0.406],
                 stds=[0.229, 0.224, 0.225],
                 n_tta=1,
                 ):
        self.df = df
        self.img_size = img_size
        self.image_path = image_path
        self.transforms = transforms
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.id_colname = id_colname
        self.img_type = img_type
        self.crop_rate = crop_rate
        self.n_tta = n_tta
        self.transforms2 = Compose([
            #CenterCrop(512 - 50, 512 - 50, p=1.0),
            Resize(img_size, img_size, p=1)
        ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        img_path = os.path.join(self.image_path, img_id + self.img_type)
        img = cv2.imread(img_path)

        if self.transforms is not None:
            augmented = self.transforms2(image=img)
            img_tta = augmented['image']
            augmented = self.transforms(image=img)
            img = augmented['image']

        imgs = []
        img = img / 255
        img -= self.means
        img /= self.stds
        img = img.transpose((2, 0, 1))
        imgs.append(torch.FloatTensor(img))
        if self.n_tta >= 2:
            flip_img = img[:, :, ::-1].copy()
            imgs.append(torch.FloatTensor(flip_img))

        if self.n_tta >= 4:
            img_tta = img_tta / 255
            img_tta -= self.means
            img_tta /= self.stds
            img_tta = img_tta.transpose((2, 0, 1))
            imgs.append(torch.FloatTensor(img_tta))
            flip_img_tta = img_tta[:, :, ::-1].copy()
            imgs.append(torch.FloatTensor(flip_img_tta))

        return imgs


def pytorch_image_to_tensor_transform(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image).float().div(255)

    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]

    return tensor

def random_cropping(image, ratio=0.8, is_random=True):
    ratio = random.random() * (1-ratio) + ratio
    height, width, _ = image.shape
    target_h = int(height*ratio)
    target_w = int(width*ratio)

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    image = image[start_y:start_y+target_h,start_x:start_x+target_w,:]

    #zeros = cv2.resize(zeros ,(width,height)) #pad to original size
    return image


class EvenSampler(Sampler):
    def __init__(self, train_df, demand_non_empty_proba):
        assert demand_non_empty_proba > 0, 'frequensy of non-empty images must be greater then zero'
        self.positive_proba = demand_non_empty_proba

        self.train_df = train_df.reset_index(drop=True)

        self.positive_idxs = self.train_df[self.train_df.sum_target != 0].index.values
        self.negative_idxs = self.train_df[self.train_df.sum_target == 0].index.values

        self.n_positive = self.positive_idxs.shape[0]
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)
        LOGGER.info("len data = {}".format(self.n_positive + self.n_negative))

    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative
