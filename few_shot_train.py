# few_shot_train.py
import numpy as np
import torch
import torch.optim as optim
from ensemble.meta_head import MetaHead

def train_meta_head(support_embeddings, support_labels, val_embeddings=None, val_labels=None,
                    input_dim=None, lr=1e-3, epochs=200, device="cuda"):
    """
    support_embeddings: (Ns, D) numpy
    support_labels: (Ns,) binary {0(real),1(fake)}
    Optionally returns trained model (torch)
    """

    if input_dim is None:
        input_dim = support_embeddings.shape[1]

    model = MetaHead(input_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    X = torch.from_numpy(support_embeddings).float().to(device)
    y = torch.from_numpy(support_labels).float().to(device)

    for ep in range(epochs):
        model.train()
        logits = model(X)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep+1) % 50 == 0:
            print(f"Epoch {ep+1}/{epochs} loss={loss.item():.4f}")

    return model
