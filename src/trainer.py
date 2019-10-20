import gc
import numpy as np
import torch
from logger import LOGGER


def train_one_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=1,
                    steps_upd_logging=1000, save_step=500, eps=1e-6):
    model.train()

    total_loss = 0.0
    for step, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, targets)

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))

    return total_loss / (step + 1)


def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    true_ans_list = []
    preds_cat = []
    for step, (features, targets) in enumerate(val_loader):
        features, targets = features.to(device), targets.to(device)

        logits = model(features)
        loss = criterion(logits, targets)

        total_loss += loss.item()

        targets = targets.float().cpu().detach().numpy().astype("int8")
        logits = torch.sigmoid(logits).float().cpu().detach().numpy().astype("float32")
        true_ans_list.append(targets)
        preds_cat.append(logits)

        del targets, logits
        gc.collect()

    all_true_ans = np.concatenate(true_ans_list, axis=0)
    all_preds = np.concatenate(preds_cat, axis=0)

    return all_preds, all_true_ans, total_loss / (step + 1)


def predict(model, test_loader, device, n_tta=1):
    model.eval()

    preds_cat = []

    with torch.no_grad():
        for step, imgs in enumerate(test_loader):
            features = imgs[0].to(device)


            logits = model(features)
            if n_tta >= 2:
                flip_img = imgs[1].to(device)
                logits += model(flip_img)
                del flip_img
                gc.collect()

            if n_tta >= 4:
                img_tta = imgs[2].to(device)
                logits += model(img_tta)
                del img_tta
                gc.collect()

                img_tta_flip = imgs[3].to(device)
                logits += model(img_tta_flip)
                del img_tta_flip
                gc.collect()

            logits = logits / n_tta

            del imgs
            gc.collect()

            logits = torch.sigmoid(logits).float().detach().cpu().numpy()
            preds_cat.append(logits)

        all_preds = np.concatenate(preds_cat, axis=0)


    return all_preds
