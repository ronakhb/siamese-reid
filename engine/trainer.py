# encoding: utf-8

import os
import torch
import datetime
import shutil
import random
import cv2
import numpy as np
import copy
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from engine.inference import inference
from engine.inference import inference_path, inference_base, inference_movie_aligned
from engine.inference import inference_prcc_global

from utils.iotools import AverageMeter
import copy
import errno
import shutil
from PIL import Image
from tqdm import tqdm
import wandb
import torch.nn as nn
import torch.optim as optim


def fliplr(img):
    # flip horizontal
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.cpu().index_select(3, inv_idx)
    return img_flip.cuda()


def norm(f):
    f = f.squeeze()
    fnorm = torch.norm(f, p=2, dim=1, keepdim=True)
    f = f.div(fnorm.expand_as(f))
    return f


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def normalize_tensor(x):      # [64, 1, 28, 28]
    map_size = x.size()
    aggregated = x.view(map_size[0], map_size[1], -1)      # [64, 1, 784]
    minimum, _ = torch.min(aggregated, dim=-1, keepdim=True)             # [64, 1, 1]
    maximum, _ = torch.max(aggregated, dim=-1, keepdim=True)             # [64, 1, 1]
    normalized = torch.div(aggregated - minimum, maximum - minimum)      # [64, 1, 784]
    normalized = normalized.view(map_size)      # [64, 1, 28, 28]

    return normalized

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])
        return old_lr

def do_train_last(cfg, model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn, num_query, num_query_test, start_epoch, acc_best):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0
    print_num = int(len(train_loader) / 5)
    if(cfg.log_wandb==1):
        now = datetime.datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
        run_name = f"MobileNet_LaST_{cfg.max_epochs}_epochs__{dt_string}"
        wandb_run = wandb.init(
            project="Mobilenet ReID LaST Script", config=vars(cfg), name=run_name
        )

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()

        for ii, input in enumerate(tqdm(train_loader)):       # [64, 3, 256, 128],  [64,],  len(train_loader)=980
            img, target, path = input
            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]

            scores, feats = model(img)                # [64, 5000], [64, 2048]

            loss = loss_fn(scores, feats, target)
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if(cfg.log_wandb==1):
                wandb.log(
                {
                    "Training Loss": loss.item(),
                }
            )
            # compute acc
            acc = (scores.max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii + 1, len(train_loader), loss, acc, scheduler.get_last_lr()[0]))

        mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, val_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
        print(line)
        if(cfg.log_wandb==1):
                wandb.log(
                {
                    "cmc1": cmc1,
                    "cmc5": cmc5,
                    "cmc10": cmc10,
                    "cmc20": cmc20,
                    "mAP": mAP
                }
            )
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': 1,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            print("Saving Checkpoint")
            last_acc_val = acc_test

        # lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr = scheduler.get_last_lr()[0]

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('lr', float(lr), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'])
    mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, test_loader, num_query_test)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()


def do_train_market(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0
    print_num = int(len(train_loader) / 5)
    if(cfg.log_wandb==1):
        now = datetime.datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
        if(len(cfg.run_name)==0):
            run_name = f"SiameseNet_{cfg.dataset}_{cfg.max_epochs}_epochs_{dt_string}"
        else:
            run_name = f"{cfg.run_name}_{cfg.max_epochs}_epochs_{dt_string}"
        wandb_run = wandb.init(
            project="Mobilenet ReID LaST Script", config=vars(cfg), name=run_name
        )

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()

        for ii, input in enumerate(tqdm(train_loader)):               # [64, 3, 256, 128],  [64,],  len(train_loader)=980
            img, target, path = input
            optimizer.zero_grad()

            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]

            scores, feats = model(img)                          # [64, 134], [64, 2048]

            loss = loss_fn(scores, feats, target)               # 9.0224
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (scores.max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii + 1, len(train_loader), loss, acc, scheduler.get_last_lr()[0]))
                if(cfg.log_wandb==1):
                    wandb.log(
                    {
                        "Training Loss": loss, "Training Accuracy": acc
                    }
                )
        mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, test_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
        if(cfg.log_wandb==1):
                wandb.log(
                {
                    "cmc1": cmc1,
                    "cmc5": cmc5,
                    "cmc10": cmc10,
                    "cmc20": cmc20,
                    "mAP": mAP
                }
            )
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            print("Saving Checkpoint")
            last_acc_val = acc_test

        # lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr = scheduler.get_last_lr()[0]

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('lr', float(lr), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'],strict=False)
    mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()


def do_train_siamese_classifier(cfg, model, train_loader, test_loader, optimizer, scheduler, loss_fn, num_query, start_epoch, acc_best):
    writer = SummaryWriter(log_dir=cfg.logs_dir)
    use_cuda = torch.cuda.is_available()
    last_acc_val = acc_best
    loss = 0.0
    print_num = int(len(train_loader) / 5)
    if(cfg.log_wandb==1):
        now = datetime.datetime.now()
        dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
        if(len(cfg.run_name)==0):
            run_name = f"SiameseNet_{cfg.dataset}_{cfg.max_epochs}_epochs_{dt_string}"
        else:
            run_name = f"{cfg.run_name}_{cfg.max_epochs}_epochs_{dt_string}"
        wandb_run = wandb.init(
            project="Mobilenet ReID LaST Script", config=vars(cfg), name=run_name
        )

    for epoch in range(start_epoch, cfg.max_epochs):            # 120
        model.train()

        for ii, input in enumerate(tqdm(train_loader)):               # [64, 3, 256, 128],  [64,],  len(train_loader)=980
            img, target, path = input
            # print(target)
            optimizer.zero_grad()

            img = img.cuda() if use_cuda else img               # [64, 3, 256, 128]
            target = target.cuda() if use_cuda else target      # [64,]

            scores, feats = model(img)                          # [64, 134], [64, 2048]
            # print(scores)

            loss = loss_fn(scores, feats, target)               # 9.0224
            loss.backward()
            optimizer.step()

            # compute acc
            acc = (scores.max(1)[1] == target).float().mean()
            loss = float(loss)
            acc = float(acc)
            if ii % print_num == 0:
                start_time = datetime.datetime.now()
                start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
                print('{} - Train: epoch: {}  {}/{}  Loss: {:.04f}  Acc: {:.1%}  Lr: {:.2e}'.format(start_time, epoch, ii + 1, len(train_loader), loss, acc, scheduler.get_last_lr()[0]))
                if(cfg.log_wandb==1):
                    wandb.log(
                    {
                        "Training Loss": loss, "Training Accuracy": acc
                    }
                )
        mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, test_loader, num_query)
        start_time = datetime.datetime.now()
        start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
        line = '{} - cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
        if(cfg.log_wandb==1):
                wandb.log(
                {
                    "cmc1": cmc1,
                    "cmc5": cmc5,
                    "cmc10": cmc10,
                    "cmc20": cmc20,
                    "mAP": mAP
                }
            )
        print(line)
        f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
        f.write(line)
        f.close()

        # deep copy the model
        acc_test = 0.5 * (cmc1 + mAP)
        is_best = acc_test >= last_acc_val
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_acc': acc_test,
        }, is_best, fpath=cfg.logs_dir)
        if is_best:
            print("Saving Checkpoint")
            last_acc_val = acc_test

        # lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr = scheduler.get_last_lr()[0]

        writer.add_scalar('train_loss', float(loss), epoch + 1)
        writer.add_scalar('lr', float(lr), epoch + 1)
        writer.add_scalar('test_rank1', float(cmc1), epoch + 1)
        writer.add_scalar('test_mAP', float(mAP), epoch + 1)

        scheduler.step()

    # Test
    last_model_wts = torch.load(os.path.join(cfg.logs_dir, 'checkpoint_best.pth'))
    model.load_state_dict(last_model_wts['state_dict'],strict=False)
    mAP, cmc1, cmc5, cmc10, cmc20 = inference(model, test_loader, num_query)

    start_time = datetime.datetime.now()
    start_time = '%4d:%d:%d-%2d:%2d:%2d' % (start_time.year, start_time.month, start_time.day, start_time.hour, start_time.minute, start_time.second)
    line = '{} - Final: cmc1: {:.1%} cmc5: {:.1%} cmc10: {:.1%} cmc20: {:.1%} mAP: {:.1%}\n'.format(start_time, cmc1, cmc5, cmc10, cmc20, mAP)
    print(line)
    f = open(os.path.join(cfg.logs_dir, 'logs.txt'), 'a')
    f.write(line)
    f.close()

def save_checkpoint(state, is_best, fpath):
    if len(fpath) != 0:
        mkdir_if_missing(fpath)

    fpath = os.path.join(fpath, 'checkpoint.pth')
    torch.save(state, fpath, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copy(fpath, os.path.join(os.path.dirname(fpath), 'checkpoint_best.pth'))


