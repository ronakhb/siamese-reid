# encoding: utf-8


import torch

def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        if "bias" in key:
            lr = cfg.lr * cfg.bias_lr_factor
            weight_decay = cfg.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        # params += [{"params": [value], "lr": cfg.lr}]
        # optimizer = torch.optim.SGD(model.parameters(),lr = cfg.lr, momentum=cfg.momentum)
    if cfg.optimizer_name == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params, momentum=cfg.momentum)
    else:
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params)
    return optimizer