# encoding: utf-8

import numpy as np
import torch
import pandas as pd
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reset()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self,use_cosine=False,use_correlation=False):
        feats = torch.cat(self.feats, dim=0)       # [19281, 2048]
        if self.feat_norm == 'yes':
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]                           # [3368, 2048]
        q_pids = np.asarray(self.pids[:self.num_query])       # [3368,]
        q_camids = np.asarray(self.camids[:self.num_query])   # [3368,]
        # gallery
        gf = feats[self.num_query:]                           # [15913, 2048]
        g_pids = np.asarray(self.pids[self.num_query:])       # [15913,]
        g_camids = np.asarray(self.camids[self.num_query:])   # [15913,]
        m, n = qf.shape[0], gf.shape[0]
        if use_correlation:
            print("using cosine loss")
            qf_mean = torch.mean(qf, dim=1, keepdim=True)
            gf_mean = torch.mean(gf, dim=1, keepdim=True)
            qf_centered = qf - qf_mean
            gf_centered = gf - gf_mean
            qf_std = torch.std(qf, dim=1, keepdim=True)
            gf_std = torch.std(gf, dim=1, keepdim=True)
            distmat = -torch.matmul(qf_centered, gf_centered.t()) / (qf_std * gf_std.t())
        elif use_cosine:
            distmat = 1 - torch.matmul(qf, gf.t())
        else:
            distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                    torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            # distmat.addmm_(1, -2, qf, gf.t())                     # [3368, 15913]
            distmat = distmat - 2 * torch.matmul(qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP

