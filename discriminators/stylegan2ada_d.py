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

        # ---- Hook Feature Extraction ----
        self._feature = None
        self._register_feature_hook()

    def _register_feature_hook(self):
        """Attach hook to the last linear layer BEFORE the final output."""
        for name, module in self.model.named_modules():
            if "fc" in name and hasattr(module, "weight"):
                module.register_forward_hook(self._save_feature_hook)
                print(f"[HOOK] Attached to StyleGAN2-ADA penultimate layer:", name)
                break

    def _save_feature_hook(self, module, input, output):
        self._feature = input[0].detach()

    def preprocess(self, pil_image):
        img = pil_image.resize((1024, 1024))
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)
        arr = arr / 127.5 - 1
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(self.device)

    def score(self, pil_image):
        x = self.preprocess(pil_image)
        c_dim = getattr(self.model, "c_dim", 0)
        c = torch.zeros((1, c_dim), device=self.device) if c_dim else None
        with torch.no_grad():
            out = self.model(x, c)
        return float(out.item())

    def extract_features(self, pil_image):
        x = self.preprocess(pil_image)
        c_dim = getattr(self.model, "c_dim", 0)
        c = torch.zeros((1, c_dim), device=self.device) if c_dim else None

        self._feature = None
        with torch.no_grad():
            _ = self.model(x, c)

        if self._feature is None:
            raise RuntimeError("StyleGAN2-ADA feature hook failed!")

        return self._feature.view(1, -1).cpu().numpy()
