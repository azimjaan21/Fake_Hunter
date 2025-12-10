# ensemble/embedder.py

import numpy as np
from .ensemble_models import sigmoid


class EnsembleEmbedder:
    """
    Build embedding vectors from the detectors.

    Priority:
      1) If a detector implements .extract_features(pil_image) -> feature vec,
         we use that high-dimensional representation.

      2) Otherwise, we fall back to a single scalar:
         calibrated P(fake) using the same logic as DeepHunterEnsemble.

    Final embedding for one image = concatenation of per-model features:
        [feat_model1 || feat_model2 || ...]  --> 1D numpy array.
    """

    def __init__(self, models):
        """
        models: list of detector objects (StyleGAN3-D, StyleGAN2-ADA-D, VAE, ...)
        Each must have:
            - .name  (str)
            - .score(pil_image) -> logit/number
        Optionally:
            - .extract_features(pil_image) -> 1D or 2D feature array
        """
        self.models = models

    # ------------------------------------------------------------------
    # Helpers for fallback (scalar) features
    # ------------------------------------------------------------------
    def _infer_model_type(self, name: str):
        """Infer detector type from its name string."""
        n = name.lower()
        if any(k in n for k in ["stylegan", "gan_d", "progan", "diffusion"]):
            return "gan_d"
        if "vae" in n:
            return "vae"
        return "classifier"

    def _logit_to_prob_fake(self, logit, model_type: str):
        """
        Same calibration as DeepHunterEnsemble:
        logit -> P(fake).
        """
        if model_type == "gan_d":
            # GAN D: high score = REAL -> flip sign
            s = -logit

        elif model_type == "vae":
            # Some VAE detectors may output [real_logit, fake_logit]
            if isinstance(logit, (list, tuple, np.ndarray)) and len(logit) == 2:
                s = logit[1] - logit[0]
            else:
                s = logit
        else:
            # classifier: positive = fake
            s = logit

        s = float(np.clip(s, -20, 20))
        return float(sigmoid(s))

    # ------------------------------------------------------------------
    # Core feature extraction
    # ------------------------------------------------------------------
    def _extract_model_feature(self, model, pil_image):
        """
        Extract feature vector for a single model.

        If model has .extract_features -> use it (high-dimensional).
        Else -> use single scalar P(fake) as a 1D feature.
        """
        # ---- Case 1: model provides high-dimensional features ----
        if hasattr(model, "extract_features"):
            feat = model.extract_features(pil_image)

            # Convert to numpy and flatten to 1D
            if isinstance(feat, np.ndarray):
                vec = feat
            else:
                # e.g., a torch tensor
                try:
                    vec = feat.detach().cpu().numpy()
                except Exception:
                    vec = np.array(feat, dtype=np.float32)

            vec = np.asarray(vec, dtype=np.float32).reshape(-1)
            # L2 normalize for ProtoNet stability
            norm = np.linalg.norm(vec) + 1e-10
            vec = vec / norm
            return vec

        # ---- Case 2: fallback to scalar P(fake) ----
        raw = model.score(pil_image)
        m_type = self._infer_model_type(model.name)
        p_fake = self._logit_to_prob_fake(raw, m_type)
        return np.array([p_fake], dtype=np.float32)

    def extract_single(self, pil_image):
        """
        Run all detectors on ONE PIL image and return concatenated feature vector.

        Returns:
            1D numpy array of shape (D_total,)
        """
        feat_list = []
        for m in self.models:
            vec = self._extract_model_feature(m, pil_image)
            feat_list.append(vec)

        # Concatenate along feature dimension
        return np.concatenate(feat_list, axis=0).astype(np.float32)

    def build_matrix(self, images):
        """
        images: list of PIL images.
        Returns:
            E: (num_images, D_total) feature matrix.
        """
        feats = [self.extract_single(img) for img in images]
        return np.stack(feats, axis=0)