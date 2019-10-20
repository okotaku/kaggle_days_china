# ===============
# SeResNext
# ===============
import os
import gc
import time
import pandas as pd
import numpy as np
from contextlib import contextmanager
from albumentations import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader

import sys

sys.path.append("../input/src-kaggledays/")
from util_tools import seed_torch
from model import CnnModel
from datasets import KDDataset, KDDatasetTest
from logger import setup_logger, LOGGER
from trainer import train_one_epoch, validate, predict

# ===============
# Constants
# ===============
DATA_DIR = "../input/kaggledays-china/"
IMAGE_PATH = os.path.join(DATA_DIR, "train3c/train3c")
TEST_IMAGE_PATH = os.path.join(DATA_DIR, "test3c/test3c/test3c_data")
LOGGER_PATH = "log.txt"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
ID_COLUMNS = "id"
# TARGET_COLUMNS = ["is_star"]
TARGET_COLUMNS = ["is_star", "loc_x", "loc_y"]
N_CLASSES = len(TARGET_COLUMNS)

# ===============
# Settings
# ===============
SEED = np.random.randint(100000)
device = "cuda"
img_size = 224
batch_size = 128
fold_id = 0
epochs = 20
EXP_ID = "exp4_seresnext"
model_path = None
save_path = '{}_fold{}.pth'.format(EXP_ID, fold_id)

setup_logger(out_file=LOGGER_PATH)
seed_torch(SEED)
LOGGER.info("seed={}".format(SEED))


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


def main():
    with timer('load data'):
        df = pd.read_csv(TRAIN_PATH)
        df["loc_x"] = df["loc_x"] / 100
        df["loc_y"] = df["loc_y"] / 100
        y = df[TARGET_COLUMNS].values
        df = df[[ID_COLUMNS]]
        gc.collect()

    with timer("split data"):
        if y.shape[1] == 1:
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(df, y)
        else:
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(df, y[:, 0])
        for n_fold, (train_index, val_index) in enumerate(folds):
            train_df = df.loc[train_index]
            val_df = df.loc[val_index]
            y_train = y[train_index]
            y_val = y[val_index]
            if n_fold == fold_id:
                break

    with timer('preprocessing'):
        train_augmentation = Compose([
            Flip(p=0.5),
            OneOf([
                ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                GridDistortion(p=0.5),
                OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            ], p=0.5),
            RandomBrightnessContrast(p=0.5),
            ShiftScaleRotate(rotate_limit=20, p=0.5),
            Resize(img_size, img_size, p=1)
        ])
        val_augmentation = Compose([
            Resize(img_size, img_size, p=1)
        ])

        train_dataset = KDDataset(train_df, y_train, img_size, IMAGE_PATH, id_colname=ID_COLUMNS,
                                  transforms=train_augmentation)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        val_dataset = KDDataset(val_df, y_val, img_size, IMAGE_PATH, id_colname=ID_COLUMNS,
                                transforms=val_augmentation)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        del df, train_dataset, val_dataset
        gc.collect()

    with timer('create model'):
        model = CnnModel(num_classes=N_CLASSES, encoder="se_resnext50_32x4d",
                         pretrained="../input/pytorch-pretrained-models/se_resnext50_32x4d-a260b3a4.pth",
                         pool_type="avg")
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.to(device)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-4)

        # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    with timer('train'):
        best_score = 0
        best_epoch = 0
        for epoch in range(1, epochs + 1):
            seed_torch(SEED + epoch)

            if epoch == epochs - 3:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

            LOGGER.info("Starting {} epoch...".format(epoch))
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, N_CLASSES)
            LOGGER.info('Mean train loss: {}'.format(round(tr_loss, 5)))

            y_pred, target, val_loss = validate(model, val_loader, criterion, device, N_CLASSES)
            score = roc_auc_score(target, y_pred)
            LOGGER.info('Mean val loss: {}'.format(round(val_loss, 5)))
            LOGGER.info('val score: {}'.format(round(score, 5)))

            if score > best_score:
                best_score = score
                best_epoch = epoch
                np.save("y_pred.npy", y_pred)
                torch.save(model.state_dict(), save_path)

        np.save("target.npy", target)
        LOGGER.info('best score: {} on epoch: {}'.format(round(best_score, 5), best_epoch))

    with timer('predict'):
        test_df = pd.read_csv(TEST_PATH)
        test_ids = test_df["id"].values

        test_augmentation = Compose([
            Resize(img_size, img_size, p=1)
        ])
        test_dataset = KDDatasetTest(test_df, img_size, TEST_IMAGE_PATH, id_colname=ID_COLUMNS,
                                     transforms=test_augmentation, n_tta=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model.load_state_dict(torch.load(save_path))

        pred = predict(model, test_loader, device, N_CLASSES, n_tta=2)
        print(pred.shape)
        results = pd.DataFrame({"id": test_ids,
                                "is_star": pred.reshape(-1)})

        results.to_csv("results.csv", index=False)


if __name__ == '__main__':
    main()
