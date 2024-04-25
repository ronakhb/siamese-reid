# encoding: utf-8

from .siamese_net import SiameseTwin,SiameseTwinSmall
from .osnet import osnet_x0_25,osnet_x0_5,osnet_x0_75,osnet_x1_0,osnet_ibn_x1_0
from .osnet_ain import osnet_ain_x0_25,osnet_ain_x0_5,osnet_ain_x0_75,osnet_ain_x1_0
from .baseline import Baseline


def build_model(num_classes=None, model_type='siamese', pretrain_choice=True,feature_dim=1000):
    if model_type == "siamese":
        model = SiameseTwin(num_classes,pretrain_choice)
    elif model_type == "siamese_small":
        model = SiameseTwinSmall(num_classes,pretrain_choice)
    elif model_type == "osnet_x0_25":
        model = osnet_x0_25(num_classes,pretrain_choice,loss='triplet',feature_dim=feature_dim)
    elif model_type == "osnet_x0_5":
        model = osnet_x0_5(num_classes,pretrain_choice,loss='triplet',feature_dim=feature_dim)
    elif model_type == "osnet_x0_75":
        model = osnet_x0_75(num_classes,pretrain_choice,loss='triplet',feature_dim=feature_dim)
    elif model_type == "osnet_x1_0":
        model = osnet_x1_0(num_classes,pretrain_choice,loss='triplet',feature_dim=feature_dim)
    elif model_type == "osnet_ibn_x1_0":
        model = osnet_ibn_x1_0(num_classes,pretrain_choice,loss='triplet',feature_dim=feature_dim)
    elif model_type == "osnet_ain_x0_25":
        model = osnet_ain_x0_25(num_classes,pretrain_choice,loss='triplet',feature_dim=feature_dim)
    elif model_type == "osnet_ain_x0_5":
        model = osnet_ain_x0_5(num_classes,pretrain_choice,loss='triplet',feature_dim=feature_dim)
    elif model_type == "osnet_ain_x0_75":
        model = osnet_ain_x0_75(num_classes,pretrain_choice,loss='triplet',feature_dim=feature_dim)
    elif model_type == "osnet_ain_x1_0":
        model = osnet_ain_x1_0(num_classes,pretrain_choice,loss='triplet',feature_dim=feature_dim)
    elif model_type == "baseline":
        model = Baseline(num_classes,pretrain_choice)
    else:
        pass
    return model