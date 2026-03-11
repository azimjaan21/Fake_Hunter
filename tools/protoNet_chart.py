import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# Reproducibility
# ============================
np.random.seed(42)

# ============================
# Generate unified embedding Z
# ============================
n_real = 200
n_fake = 200

real_embeddings = np.random.multivariate_normal(
    mean=[0.4, 0.6],
    cov=[[0.02, 0], [0, 0.02]],
    size=n_real
)

fake_embeddings = np.random.multivariate_normal(
    mean=[0.7, 0.3],
    cov=[[0.02, 0], [0, 0.02]],
    size=n_fake
)

# ============================
# ProtoNet prototypes
# ============================
real_proto = real_embeddings.mean(axis=0)
fake_proto = fake_embeddings.mean(axis=0)

# ============================
# Query embedding
# ============================
query = np.array([0.55, 0.45])

# ============================
# Plot
# ============================
plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")

# Image embeddings (background)
sns.scatterplot(
    x=real_embeddings[:, 0],
    y=real_embeddings[:, 1],
    color="tab:green",
    alpha=0.3,
    s=40,
    label="Real embeddings"
)

sns.scatterplot(
    x=fake_embeddings[:, 0],
    y=fake_embeddings[:, 1],
    color="tab:red",
    alpha=0.3,
    s=40,
    label="Fake embeddings"
)

# Prototypes
plt.scatter(
    real_proto[0], real_proto[1],
    marker="*", s=300, color="green",
    edgecolor="black", label="Real prototype"
)

plt.scatter(
    fake_proto[0], fake_proto[1],
    marker="*", s=300, color="red",
    edgecolor="black", label="Fake prototype"
)

# Query embedding
plt.scatter(
    query[0], query[1],
    s=200, color="blue",
    edgecolor="black", label="Query embedding"
)

# Distance lines
plt.plot(
    [query[0], real_proto[0]],
    [query[1], real_proto[1]],
    linestyle="--", color="green"
)

plt.plot(
    [query[0], fake_proto[0]],
    [query[1], fake_proto[1]],
    linestyle="--", color="red"
)

# ============================
# Labels & formatting
# ============================
plt.title("ProtoNet Inference in Unified Embedding Space Z", fontsize=14)
plt.xlabel("Embedding Dimension 1")
plt.ylabel("Embedding Dimension 2")

plt.legend(loc="best", frameon=True)
plt.axis("equal")
plt.tight_layout()

plt.show()
