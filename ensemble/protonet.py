# ensemble/protonet.py

import numpy as np

def build_prototypes(E, y):
    """
    E: (N, D) embedding matrix
    y: list/array of labels, e.g. [0,0,0,1,1,1]  (0=real, 1=fake)

    Returns:
        prototypes: (C, D) matrix (mean for each class)
        classes:    (C,) array of unique class labels (sorted)
    """
    E = np.asarray(E, dtype=np.float32)
    y = np.asarray(y)

    classes = np.unique(y)
    protos = []
    for c in classes:
        protos.append(E[y == c].mean(axis=0))

    prototypes = np.stack(protos, axis=0)   # (C, D)
    return prototypes, classes


def _cosine_similarity(a, b):
    """
    a: (1, D)
    b: (C, D)
    return: (C,) cosine similarity
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.ndim == 1:
        a = a[None, :]

    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

    # (1, D) @ (D, C) → (1, C)
    sim = a_norm @ b_norm.T
    return sim[0]  # (C,)


def predict_with_prototypes(q, prototypes, metric="cosine"):
    """
    q: (1, D) or (D,) query embedding
    prototypes: (C, D)
    metric: "cosine" for now

    Returns:
        probs:  (1, C) softmax over similarities
        logits: (1, C) raw similarity scores
    """
    if metric == "cosine":
        sim = _cosine_similarity(q, prototypes)  # (C,)
        logits = sim[None, :]                    # (1, C)
    else:
        raise NotImplementedError("Only cosine metric supported for now.")

    # softmax over classes
    logits_shift = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits_shift)
    probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-8)

    return probs, logits
