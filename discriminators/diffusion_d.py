import numpy as np
import torch
from .base_discriminator import BaseDiscriminator
import legacy

class DiffusionStyleGAN2Discriminator(BaseDiscriminator):
    def __init__(self, checkpoint):
        super().__init__("Diffusion-SG2-D")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Loading DiffusionGAN Discriminator: {checkpoint}")
        with open(checkpoint, "rb") as f:
            self.model = legacy.load_network_pkl(f)['D'].eval().to(self.device)

    def preprocess(self, pil_image):
        img = pil_image.resize((1024, 1024))
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
        arr = arr / 127.5 - 1
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor

    def score(self, pil_image):
        x = self.preprocess(pil_image)
        with torch.no_grad():
            out = self.model(x)
        return float(out.item())
