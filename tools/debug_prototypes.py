import pickle
import numpy as np

proto_path = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\Fake_Hunter\fewshot_cache\prototypes.pkl"
embed_path = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\Fake_Hunter\fewshot_cache\support_embeddings.pkl"

print("\n=== Checking Prototype File ===")
with open(proto_path, "rb") as f:
    proto = pickle.load(f)

print("Classes:", proto["classes"])
print("Prototype shape:", proto["prototypes"].shape)

print("\n=== Checking Support Embeddings ===")
with open(embed_path, "rb") as f:
    emb = pickle.load(f)

E = emb["E"]
y = emb["y"]

print("Support count:", len(y))
print("Real count:", y.count(0))
print("Fake count:", y.count(1))
print("Embedding shape:", E.shape)

# Distance check
from numpy.linalg import norm

real_proto = proto["prototypes"][0]
fake_proto = proto["prototypes"][1]

dist = norm(real_proto - fake_proto)
print("\nDistance between REAL and FAKE prototypes:", dist)
