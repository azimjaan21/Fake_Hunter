# ensemble/ensemble_models.py
import numpy as np

def sigmoid(x):
    # Numerically stable sigmoid function
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

class DeepHunterEnsemble:
    def __init__(self, config):
        """
        Initializes the ensemble with model configurations.

        config: List of dicts, example:
        [
            {'name': 'stylegan3', 'type': 'gan_d', 'weight': 1.0},
            {'name': 'stylegan2ada', 'type': 'gan_d', 'weight': 1.0},
            {'name': 'vae', 'type': 'vae', 'weight': 1.0}
        ]
        """

        self.models = config
        self.n = len(config)

    def _process_single_logit(self, logit, model_type):
        """
        Normalize and convert raw logits → P(fake)
        Supports: gan_d, classifier, vae
        """

        # ------------------------------
        # 1. Polarity Fix
        # ------------------------------
        if model_type == 'gan_d':
            # GAN discriminator returns high score = real → invert
            s = -logit

        elif model_type == 'vae':
            # Some VAE detectors output 2 logits: real/fake
            # If logit is a list/tuple of size 2, convert properly
            if isinstance(logit, (list, tuple, np.ndarray)) and len(logit) == 2:
                # logit[0] = real, logit[1] = fake
                s = logit[1] - logit[0]  # Convert to “fake logit”
            else:
                # If single logit provided, treat it normally
                s = logit

        else:
            # Classifier: logit > 0 = fake
            s = logit

        # ------------------------------
        # 2. Calibration
        # ------------------------------
        s = np.clip(s, -20, 20)
        return sigmoid(s)

    def predict(self, raw_logits):
        """
        raw_logits: list of logits, in same order as config
        Returns: dict with ensemble results
        """

        raw_logits = np.array(raw_logits, dtype=np.float32)
        probs = []

        # Convert each raw detector output → P(fake)
        for i, logit in enumerate(raw_logits):
            if i >= len(self.models):
                break

            model_type = self.models[i].get("type", "classifier")
            p_fake = self._process_single_logit(logit, model_type)
            probs.append(p_fake)

        probs = np.array(probs)

        # ---------------------------------------
        # 2. Aggregation
        # ---------------------------------------
        weights = np.array([m.get("weight", 1.0) for m in self.models])
        weights = weights[:len(probs)]
        weights /= (weights.sum() + 1e-9)

        weighted_avg = np.sum(probs * weights)
        median_prob = np.median(probs)
        vote_ratio = np.mean((probs > 0.5).astype(np.float32))
        max_suspicion = np.max(probs)

        # ---------------------------------------
        # 3. Final Robust Fusion
        # ---------------------------------------
        final_score = weighted_avg

        if self.n <= 3:
            # For small ensembles, median stabilizes outliers
            final_score = float(np.mean([weighted_avg, median_prob]))
        else:
            # Larger ensemble: boost if strong agreement
            if vote_ratio > 0.7:
                final_score = (final_score + max_suspicion) / 2

        # ---------------------------------------
        return {
            "per_model_probs": np.round(probs, 4).tolist(),
            "avg_prob": float(np.mean(probs)),
            "weighted_prob": float(weighted_avg),
            "median_prob": float(median_prob),
            "final_prob": float(final_score),
            "vote_ratio": float(vote_ratio),
        }
