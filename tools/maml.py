import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# =========================
# Generate unified space Z
# =========================
np.random.seed(0)

real = np.random.multivariate_normal([0.4, 0.6], [[0.02,0],[0,0.02]], 200)
fake = np.random.multivariate_normal([0.7, 0.3], [[0.02,0],[0,0.02]], 200)

X = np.vstack([real, fake])
y = np.array([0]*len(real) + [1]*len(fake))  # 0=Real, 1=Fake

# =========================
# MAML-style adapted head
# (nonlinear decision boundary)
# =========================
clf = SVC(kernel="rbf", gamma=8, C=1.0)
clf.fit(X, y)

# =========================
# Plot
# =========================
plt.figure(figsize=(7,7))
sns.set(style="whitegrid")

sns.scatterplot(x=real[:,0], y=real[:,1],
                color="green", alpha=0.3, label="Real")

sns.scatterplot(x=fake[:,0], y=fake[:,1],
                color="red", alpha=0.3, label="Fake")

# Decision boundary
xx, yy = np.meshgrid(
    np.linspace(0,1,400),
    np.linspace(0,1,400)
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

plt.title("Fig (b): MAML — Curved Decision Boundary", fontsize=14)
plt.xlabel("Embedding Dimension 1")
plt.ylabel("Embedding Dimension 2")
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()
