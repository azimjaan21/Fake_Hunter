import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import numpy as np
import os


# ============================================================
#                   FREQUENCY ANALYZER BRANCH
# ============================================================

class FrequencyAnalyzer(nn.Module):
    """Frequency domain analysis branch using FFT + small CNN."""

    def __init__(self):
        super(FrequencyAnalyzer, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def extract_frequency_features(self, x):
        """Convert RGB → grayscale → FFT → magnitude spectrum."""
        # x: [B, 3, H, W]
        gray = x.mean(dim=1).detach().cpu().numpy()  # [B, H, W]
        batch_mag = []

        for img in gray:
            f = np.fft.fft2(img)
            f = np.fft.fftshift(f)
            mag = np.log(np.abs(f) + 1)

            mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
            batch_mag.append(mag)

        batch_mag = np.array(batch_mag)  # [B, H, W]
        freq = torch.tensor(batch_mag, dtype=torch.float32).unsqueeze(1)  # [B,1,H,W]
        return freq.to(x.device)

    def forward(self, x):
        freq = self.extract_frequency_features(x)
        feat = self.conv_layers(freq)
        return feat.view(feat.size(0), -1)


# ============================================================
#                 HYBRID CNN + FREQUENCY MODEL
# ============================================================

class HybridVAEDetector(nn.Module):
    """Classifier: ResNet18 CNN + FFT frequency CNN branch."""

    def __init__(self):
        super(HybridVAEDetector, self).__init__()

        self.cnn_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.cnn_backbone.fc.in_features
        self.cnn_backbone.fc = nn.Identity()  # remove classification head

        self.freq_analyzer = FrequencyAnalyzer()

        self.fusion = nn.Sequential(
            nn.Linear(num_features + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # [real_logit, fake_logit]
        )

    def forward(self, x):
        cnn_features = self.cnn_backbone(x)
        freq_features = self.freq_analyzer(x)
        combined = torch.cat([cnn_features, freq_features], dim=1)

        return self.fusion(combined)


# ============================================================
#                  LOAD FUNCTION (MODE-SAFE)
# ============================================================

def load_hybrid_vae(weights_path, device="cuda"):
    model = HybridVAEDetector().to(device)

    if os.path.exists(weights_path):
        print(f"🔄 Loading VAE Hybrid weights from: {weights_path}")
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)
    else:
        print(f"⚠ WARNING: Weight file not found → using untrained model: {weights_path}")

    model.eval()
    return model


def get_hybrid_model(device='cuda'):
    model = HybridVAEDetector().to(device)
    print(f"✅ HybridVAEDetector loaded on {device}")
    return model


# ============================================================
#               WRAPPER FOR ENSEMBLE COMPATIBILITY
# ============================================================

class VAEHybridDetectorWrapper:
    """Adds .score(pil_image) so this model behaves like other discriminators."""

    def __init__(self, weights_path, device="cuda"):
        self.device = device
        self.name = "VAEHybrid"
        self.type = "vae"
        self.model = load_hybrid_vae(weights_path, device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    def preprocess(self, pil_image: Image.Image):
        img = self.transform(pil_image).unsqueeze(0).to(self.device)
        return img

    def score(self, pil_image: Image.Image):
        """
        Returns: raw logit where positive → FAKE, negative → REAL.
        Compatible with existing GAN discriminator .score().
        """
        x = self.preprocess(pil_image)

        with torch.no_grad():
            logits = self.model(x).squeeze(0)  # [2]

        real_logit = float(logits[0].cpu())
        fake_logit = float(logits[1].cpu())

        return fake_logit - real_logit   # single scalar logit
