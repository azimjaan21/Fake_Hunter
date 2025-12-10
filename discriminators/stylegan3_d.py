import sys, os
import numpy as np
import torch
from .base_discriminator import BaseDiscriminator

# ============================================================
# 1. REMOVE ALL stylegan2-ada paths from sys.path
# ============================================================
sys.path = [p for p in sys.path if "stylegan2" not in p.lower()]
sys.path = [p for p in sys.path if "diffusion" not in p.lower()]

# ============================================================
# 2. Add ONLY StyleGAN3 path (and put it at highest priority)
# ============================================================
STYLEGAN3_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "external", "stylegan3")
)

sys.path.insert(0, STYLEGAN3_PATH)

print("[DEBUG] StyleGAN3 path inserted:", STYLEGAN3_PATH)
print("[DEBUG] sys.path top entries:", sys.path[:3])

# ============================================================
# 3. Import correct SG3 legacy
# ============================================================
import legacy
# ============================================================

class StyleGAN3Discriminator(BaseDiscriminator):
    def __init__(self, checkpoint):
        super().__init__("StyleGAN3-D")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Loading StyleGAN3 Discriminator: {checkpoint}")
        with open(checkpoint, "rb") as f:
            self.model = legacy.load_network_pkl(f)['D'].eval().to(self.device)

        # ---- Hook Feature Extraction ----
        self._feature = None
        self._register_feature_hook()

    def _register_feature_hook(self):
        """Hook on the layer BEFORE the final fully-connected output."""
        for name, module in self.model.named_modules():
            if "fc" in name and hasattr(module, "weight"):
                module.register_forward_hook(self._save_feature_hook)
                print(f"[HOOK] Attached to StyleGAN3 penultimate layer: {name}")
                break

    def _save_feature_hook(self, module, input, output):
        # Save feature BEFORE final classification
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
        """Return 1×C feature vector from penultimate layer."""
        x = self.preprocess(pil_image)
        c_dim = getattr(self.model, "c_dim", 0)
        c = torch.zeros((1, c_dim), device=self.device) if c_dim else None

        self._feature = None
        with torch.no_grad():
            _ = self.model(x, c)

        if self._feature is None:
            raise RuntimeError("StyleGAN3 feature hook failed!")

        return self._feature.view(1, -1).cpu().numpy()
