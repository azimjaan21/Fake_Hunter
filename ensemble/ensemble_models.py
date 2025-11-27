# ensemble/ensemble_models.py
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class SimpleEnsemble:
    """
    Simple ensemble that:
    - Takes raw logit-like scores from each discriminator
    - Produces:
        * soft-average probability
        * weighted probability (optional)
    """

    def __init__(self, model_names):
        self.model_names = model_names
        # equal weights as default
        self.weights = np.ones(len(model_names), dtype=np.float32) / len(model_names)

    def set_weights(self, weights):
        w = np.array(weights, dtype=np.float32)
        w = w / (w.sum() + 1e-6)
        self.weights = w

    def predict_proba(self, logits):
        """
        logits: list or np.array of per-model logits
        Returns:
            dict with:
                - 'per_model_probs'
                - 'avg_prob'
                - 'weighted_prob'
        """
        logits = np.array(logits, dtype=np.float32)
        probs = sigmoid(logits)

        avg_prob = probs.mean()
        weighted_prob = (probs * self.weights).sum()

        return {
            "per_model_probs": probs.tolist(),
            "avg_prob": float(avg_prob),
            "weighted_prob": float(weighted_prob),
        }
