import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch

class Baseline(nn.Module):
    def __init__(self, num_classes, pretrain_choice=True):
        super(Baseline, self).__init__()

        # ResNet-50 as the backbone
        self.backbone = models.resnet50(pretrained=pretrain_choice)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the final classification layer

        # Classifier head with GAP and batch normalization
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.batch_norm = nn.BatchNorm1d(in_features, track_running_stats=True)
        self.linear_layer = nn.Linear(in_features, num_classes)

        self.num_classes = num_classes

    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.backbone.layer4(x)

        # Global average pooling and batch normalization
        pooled_features = self.global_pooling(features)        
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        normalized_features = self.batch_norm(pooled_features)
        
        if self.training:
            # Classification
            logits = self.linear_layer(normalized_features)
            return logits, normalized_features
        else:
            return normalized_features
