import numpy as np
import torch
from .base_discriminator import BaseDiscriminator
import legacy

class StyleGAN2ADADiscriminator(BaseDiscriminator):
    def __init__(self, checkpoint):
        super().__init__("StyleGAN2-ADA-D")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Loading StyleGAN2-ADA Discriminator: {checkpoint}")
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
        c_dim = getattr(self.model, "c_dim", 0)
        c = torch.zeros((1, c_dim), device=self.device) if c_dim is not None else None
        with torch.no_grad():
            out = self.model(x, c)
        return float(out.item())
