import torch
import torch.nn as nn


def baseline(num_classes):
    model = torch.hub.load('pytorch/vision:v0.9.0',
                           'resnet18', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  
    return model
