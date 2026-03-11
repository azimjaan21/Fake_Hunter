import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class VAEDetectionDataset(Dataset):
    """VAE 생성 이미지 탐지용 데이터셋"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # real과 vae 이미지 경로 수집
        self.real_images = self._get_images(os.path.join(data_dir, 'real'))
        self.vae_images = self._get_images(os.path.join(data_dir, 'vae'))
        
        # 전체 이미지 리스트와 레이블 생성
        self.images = self.real_images + self.vae_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.vae_images)
        
        print(f"Real 이미지: {len(self.real_images)}장")
        print(f"VAE 이미지: {len(self.vae_images)}장")
        print(f"전체: {len(self.images)}장")
    
    def _get_images(self, folder_path):
        """폴더에서 이미지 파일 경로 가져오기"""
        images = []
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append(os.path.join(folder_path, filename))
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        # 전처리 적용
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_dataloaders(batch_size=32):
    """학습/테스트 데이터로더 생성"""
    
    # 이미지 전처리 설정
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 생성
    train_dataset = VAEDetectionDataset('dataset/train', transform=transform)
    test_dataset = VAEDetectionDataset('dataset/test', transform=transform)
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    
    return train_loader, test_loader