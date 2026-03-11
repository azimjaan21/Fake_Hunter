import torch
import torch.nn as nn
from torchvision import models
import numpy as np

class FrequencyAnalyzer(nn.Module):
    """주파수 분석 브랜치"""
    
    def __init__(self):
        super(FrequencyAnalyzer, self).__init__()
        
        # 주파수 특징을 학습할 CNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
    def extract_frequency_features(self, x):
        """FFT로 주파수 도메인 변환"""
        batch_size = x.size(0)
        freq_features = []
        
        for i in range(batch_size):
            # RGB를 Grayscale로 변환
            img = x[i].mean(dim=0).cpu().numpy()
            
            # FFT 적용
            f_transform = np.fft.fft2(img)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # 로그 스케일
            magnitude = np.log(magnitude + 1)
            
            # 정규화
            magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
            
            freq_features.append(magnitude)
        
        freq_features = np.array(freq_features)
        freq_features = torch.FloatTensor(freq_features).unsqueeze(1).to(x.device)
        
        return freq_features
    
    def forward(self, x):
        # 주파수 특징 추출
        freq = self.extract_frequency_features(x)
        
        # CNN으로 주파수 특징 학습
        feat = self.conv_layers(freq)
        feat = feat.view(feat.size(0), -1)
        
        return feat


class HybridVAEDetector(nn.Module):
    """CNN + 주파수 분석 하이브리드 모델"""
    
    def __init__(self):
        super(HybridVAEDetector, self).__init__()
        
        # CNN 브랜치 (기존)
        self.cnn_backbone = models.resnet18(pretrained=True)
        num_features = self.cnn_backbone.fc.in_features
        self.cnn_backbone.fc = nn.Identity()  # FC layer 제거
        
        # 주파수 분석 브랜치 (새로 추가)
        self.freq_analyzer = FrequencyAnalyzer()
        
        # 특징 융합 및 분류
        self.fusion = nn.Sequential(
            nn.Linear(num_features + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        # CNN 특징
        cnn_feat = self.cnn_backbone(x)
        
        # 주파수 특징
        freq_feat = self.freq_analyzer(x)
        
        # 특징 결합
        combined = torch.cat([cnn_feat, freq_feat], dim=1)
        
        # 최종 분류
        output = self.fusion(combined)
        
        return output


def get_hybrid_model(device='cuda'):
    """하이브리드 모델 생성"""
    model = HybridVAEDetector()
    model = model.to(device)
    print(f"✅ 하이브리드 모델이 {device}에 로드되었습니다.")
    return model