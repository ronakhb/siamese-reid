import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

class SiameseTwin(nn.Module):
    def __init__(self, num_classes, pretrain_choice=True):
        super(SiameseTwin, self).__init__()

        # MobileNetV3 as the backbone
        if pretrain_choice == True:
            self.backbone = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.DEFAULT')
            print('Loading pretrained MobileNet model......')
        in_features = self.backbone.classifier[0].in_features
        self.linear_layer = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(in_features, num_classes))
        self.num_classes = num_classes
        self.batch_norm = nn.BatchNorm1d(in_features,track_running_stats=True)

    def forward(self, x, use_cam=False):
        embedding = self.extract_embedding(x)
        if self.training:
            prob = self.reid_classifier(embedding)         # [64, 751]
            return prob, embedding
        else:
            embedding = self.batch_norm(embedding)
            return embedding

    def extract_embedding(self, x):
        # Extract embeddings from the backbone and apply L2 normalization
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.squeeze(x,dim=2)
        x = torch.squeeze(x,dim=2)
        return x
    
    def reid_classifier(self,x):
        x = self.batch_norm(x)
        x = self.linear_layer(x)
        x = torch.sigmoid(x)
        return x

class SiameseTwinSmall(nn.Module):
    def __init__(self, num_classes, pretrain_choice=True):
        super(SiameseTwinSmall, self).__init__()

        # MobileNetV3 as the backbone
        if pretrain_choice == True:
            self.backbone = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.DEFAULT')
            print('Loading pretrained MobileNet model......')
        in_features = self.backbone.classifier[0].in_features
        self.linear_layer = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(in_features, num_classes))
        self.num_classes = num_classes
        self.batch_norm = nn.BatchNorm1d(in_features,track_running_stats=True)
        
    def forward(self, x, use_cam=False):
        embedding = self.extract_embedding(x)
        if self.training:
            prob = self.reid_classifier(embedding)         # [64, 751]
            return prob, embedding
        else:
            embedding = self.batch_norm(embedding)
            return embedding

    def extract_embedding(self, x):
        # Extract embeddings from the backbone and apply L2 normalization
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.squeeze(x)
        return x
    
    def reid_classifier(self,x):
        x = self.batch_norm(x)
        x = self.linear_layer(x)
        x = torch.sigmoid(x)
        return x

class SiameseTwinWithClassifer(nn.Module):
    def __init__(self, num_classes, pretrain_choice=True):
        super(SiameseTwinWithClassifer, self).__init__()

        # MobileNetV3 as the backbone
        if pretrain_choice == True:
            self.backbone = models.mobilenet_v3_large(weights='MobileNet_V3_Large_Weights.DEFAULT')
            print('Loading pretrained MobileNet model......')
        in_features = self.backbone.classifier[0].in_features
        self.linear_layer = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(in_features, 1))
        self.num_classes = num_classes
        self.batch_norm = nn.BatchNorm1d(in_features,track_running_stats=True)

    def forward(self, x, use_cam=False):
        embedding = self.extract_embedding(x)
        if self.training:
            prob = self.reid_classifier(embedding)         # [64, 751]
            return prob, embedding
        else:
            embedding = self.batch_norm(embedding)
            return embedding
    
    def reid_classifier(self,x1,x2):
        x = torch.cat(x1,x2)
        x = self.batch_norm(x)
        x = self.linear_layer(x)
        x = torch.sigmoid(x)
        return x