# ensemble/meta_head.py
import torch
import torch.nn as nn

class MetaHead(nn.Module):
    """
    Small MLP that takes concatenation [prototype_probs, raw_features(optional)]
    or raw embedding and outputs correction logit. Use for few-shot fine tunning.
    """

    def __init__(self, input_dim, hidden_dims=(256,64)):
        super().__init__()
        layers = []
        cur = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(0.2))
            cur = h
        layers.append(nn.Linear(cur, 1))  # output logit for FAKE
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,)
