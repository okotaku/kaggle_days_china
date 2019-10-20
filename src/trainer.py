import gc
import numpy as np
import torch
from torch.autograd import Variable
from logger import LOGGER


def get_cutmix_data(inputs, target, beta=1, device=0):
    def _rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    lam = np.random.beta(beta, beta)

    rand_index = torch.randperm(inputs.size()[0]).to(device)
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = _rand_bbox(inputs.size(), lam)

    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

    input_var = torch.autograd.Variable(inputs, requires_grad=True).to(device)
    target_a_var = torch.autograd.Variable(target_a).to(device)
    target_b_var = torch.autograd.Variable(target_b).to(device)
    return input_var, target_a_var, target_b_var, lam


def mixup_data(x, y, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, train_loader, criterion, optimizer, device, n_classes, accumulation_steps=1,
                    steps_upd_logging=1000, save_step=500, eps=1e-6, cutmix_prob=0.0, beta=1):
    model.train()

    total_loss = 0.0
    for step, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()

        if np.random.rand() < cutmix_prob:
            features, target_a_var, target_b_var, lam = get_cutmix_data(
                features.to(device),
                targets,
                beta=beta,
                device=device
            )
            logits = model(features)
            if n_classes == 1:
                loss = criterion(logits, target_a_var) * lam + criterion(logits, target_b_var) * (1. - lam)
            else:
                loss = criterion(logits[:, 0], target_a_var[:, 0]) * lam + criterion(logits[:, 0], target_b_var[:, 0]) * (1. - lam)
                loss += torch.nn.MSELoss()(logits[:, 1:], target_a_var[:, 1:]) * lam + \
                        torch.nn.MSELoss()(logits[:, 1:], target_b_var[:, 1:]) * (1. - lam)
        else:
            logits = model(features)
            if n_classes == 1:
                loss = criterion(logits, targets)
            else:
                loss = criterion(logits[:, 0], targets[:, 0])
                loss += torch.nn.MSELoss()(logits[:, 1:], targets[:, 1:])

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))

    return total_loss / (step + 1)


def train_one_epoch_mixup(model, train_loader, criterion, optimizer, device, n_classes, accumulation_steps=1,
                    steps_upd_logging=1000, save_step=500, eps=1e-6, mixup_prob=0.0, mixup_alpha=1.0):
    model.train()

    total_loss = 0.0
    for step, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()

        if np.random.rand() < mixup_prob:
            features, targets_a, targets_b, lam = mixup_data(features, targets, device, alpha=mixup_alpha)
            features, targets_a, targets_b = map(Variable, (features,
                                                          targets_a, targets_b))

            logits = model(features)
            if n_classes == 1:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                loss = mixup_criterion(criterion, logits[:, 0], targets_a[:, 0], targets_b[:, 0], lam)
                loss += mixup_criterion(torch.nn.MSELoss(), logits[:, 1:], targets_a[:, 1:], targets_b[:, 1:], lam)
        else:
            logits = model(features)
            if n_classes == 1:
                loss = criterion(logits, targets)
            else:
                loss = criterion(logits[:, 0], targets[:, 0])
                loss += torch.nn.MSELoss()(logits[:, 1:], targets[:, 1:])

        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            LOGGER.info('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))

    return total_loss / (step + 1)


def validate(model, val_loader, criterion, device, n_classes):
    model.eval()

    total_loss = 0.0
    true_ans_list = []
    preds_cat = []
    for step, (features, targets) in enumerate(val_loader):
        features, targets = features.to(device), targets.to(device)

        logits = model(features)
        if n_classes == 1:
            loss = criterion(logits, targets)
        else:
            loss = criterion(logits[:, 0], targets[:, 0])
            loss +=  torch.nn.MSELoss()(logits[:, 1:], targets[:, 1:])

        total_loss += loss.item()

        if n_classes == 1:
            targets = targets.float().cpu().detach().numpy().astype("int8")
            logits = torch.sigmoid(logits).float().cpu().detach().numpy().astype("float32")
        else:
            targets = targets[:, 0].float().cpu().detach().numpy().astype("int8")
            logits = torch.sigmoid(logits[:, 0]).float().cpu().detach().numpy().astype("float32")
        true_ans_list.append(targets)
        preds_cat.append(logits)

        del targets, logits
        gc.collect()

    all_true_ans = np.concatenate(true_ans_list, axis=0)
    all_preds = np.concatenate(preds_cat, axis=0)

    return all_preds, all_true_ans, total_loss / (step + 1)


def predict(model, test_loader, device, n_classes, n_tta=1):
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

            if n_classes == 1:
                logits = torch.sigmoid(logits).float().cpu().detach().numpy().astype("float32")
            else:
                logits = torch.sigmoid(logits[:, 0]).float().cpu().detach().numpy().astype("float32")
            preds_cat.append(logits)

        all_preds = np.concatenate(preds_cat, axis=0)


    return all_preds
