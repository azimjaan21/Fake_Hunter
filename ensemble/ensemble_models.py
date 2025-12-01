# ensemble/ensemble_models.py
import numpy as np

def sigmoid(x):
    # Numerically stable sigmoid function
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

class DeepHunterEnsemble:
    def __init__(self, config):
        """
        Initializes the ensemble with model configurations.
        config: List of dicts, e.g., [{'name': 'StyleGAN2', 'type': 'gan_d', 'weight': 1.0}, ...]
        """
        self.models = config
        self.n = len(config)
        
    def predict(self, raw_logits):
        """
        Calculates the final probability of the image being FAKE (0 to 1).
        raw_logits: List or Array of raw scores corresponding to self.models order.
        """
        raw_logits = np.array(raw_logits, dtype=np.float32).flatten()
        probs = []
        
        # 1. Process logits, fix polarity, and calibrate to P(Fake)
        for i, logit in enumerate(raw_logits):
            if i >= len(self.models): break
            
            model_conf = self.models[i]
            model_type = model_conf.get('type', 'classifier') 
            
            # -----------------------------------------------------------
            # 1a. POLARITY CORRECTION (Crucial Fix)
            # -----------------------------------------------------------
            if model_type == 'gan_d':
                # GAN Discriminators: D(x) > 0 means REAL. We want P(Fake), so we invert the sign.
                score = -logit 
            else:
                # Classifiers: Logit > 0 usually means FAKE. No inversion needed.
                score = logit

            # -----------------------------------------------------------
            # 1b. INDIVIDUAL CALIBRATION (Sigmoid)
            # -----------------------------------------------------------
            score = np.clip(score, -20, 20) # Clip for numerical stability
            p_fake = sigmoid(score)
            probs.append(p_fake)

        probs = np.array(probs)
        
        # 2. Aggregation Strategies
        weights = np.array([m.get('weight', 1.0) for m in self.models])
        if len(probs) < len(weights):
            weights = weights[:len(probs)]
            
        weights /= (weights.sum() + 1e-9)
        
        weighted_avg = np.sum(probs * weights)
        median_prob = np.median(probs) if len(probs) > 0 else 0.0
        vote_ratio = np.mean((probs > 0.5).astype(int))
        max_suspicion = np.max(probs) if len(probs) > 0 else 0.0

        # -----------------------------------------------------------
        # 3. Final Decision Logic (Robust against single outliers)
        # -----------------------------------------------------------
        
        final_score = weighted_avg # Start with the weighted average
        
        if self.n <= 3:
             # For small ensembles (like 2 models), median is more stable against one misfiring model.
             final_score = float(np.mean([weighted_avg, median_prob]))
        else:
             # For larger ensembles, boost the weighted average if there's a strong majority vote
             if vote_ratio > 0.7:
                 final_score = (final_score + max_suspicion) / 2

        return {
            "per_model_probs": np.round(probs, 4).tolist(),
            "avg_prob": float(np.mean(probs)),
            "weighted_prob": float(weighted_avg), # Weighted average (before final blend)
            "median_prob": float(median_prob),
            "final_prob": float(final_score), # The final, most robust score
            "vote_ratio": vote_ratio
        }