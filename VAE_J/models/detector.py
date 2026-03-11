import torch
import torch.nn as nn
from torchvision import models

class VAEDetector(nn.Module):
    """VAE 생성 이미지를 탐지하는 CNN 모델"""
    
    def __init__(self):
        super(VAEDetector, self).__init__()
        
        # 사전학습된 ResNet-18 사용
        self.backbone = models.resnet18(pretrained=True)
        
        # 마지막 FC layer를 2개 클래스용으로 교체
        # (원래는 1000개 ImageNet 클래스용)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 2)  # Real(0), VAE(1)
    
    def forward(self, x):
        return self.backbone(x)


def get_model(device='cuda'):
    """모델 생성 및 GPU로 이동"""
    model = VAEDetector()
    model = model.to(device)
    print(f"✅ 모델이 {device}에 로드되었습니다.")
    return model