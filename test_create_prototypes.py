import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pickle
from PIL import Image

from discriminators import get_all_discriminators
from ensemble.embedder import EnsembleEmbedder
from ensemble.protonet import build_prototypes


SUPPORT_REAL = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\Fake_Hunter\data\support\real"
SUPPORT_FAKE = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\Fake_Hunter\data\support\fake"
SAVE_DIR = r"fewshot_cache"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_images(folder):
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            files.append(os.path.join(folder, f))
    return files


if __name__ == "__main__":
    print("=== DeepHunter V2: Building Few-Shot Prototypes ===")

    real_paths = load_images(SUPPORT_REAL)
    fake_paths = load_images(SUPPORT_FAKE)

    if len(real_paths) == 0 or len(fake_paths) == 0:
        print("❌ ERROR: support/real or support/fake is empty!")
        exit()

    print(f"Loaded {len(real_paths)} REAL samples")
    print(f"Loaded {len(fake_paths)} FAKE samples")

    models = get_all_discriminators()
    embedder = EnsembleEmbedder(models)

    support_imgs = []
    labels = []

    # Load real
    for p in real_paths:
        img = Image.open(p).convert("RGB")
        support_imgs.append(img)
        labels.append(0)

    # Load fake
    for p in fake_paths:
        img = Image.open(p).convert("RGB")
        support_imgs.append(img)
        labels.append(1)

    # Step 1: Build embedding matrix
    print("Extracting embeddings...")
    E = embedder.build_matrix(support_imgs)

    # Step 2: Build prototypes
    print("Computing prototypes...")
    prototypes, classes = build_prototypes(E, labels)

    # Save everything
    with open(os.path.join(SAVE_DIR, "prototypes.pkl"), "wb") as f:
        pickle.dump({
            "prototypes": prototypes,
            "classes": classes
        }, f)

    with open(os.path.join(SAVE_DIR, "support_embeddings.pkl"), "wb") as f:
        pickle.dump({"E": E, "y": labels}, f)

    print("=== DONE ===")
    print(f"Prototypes saved to: {os.path.join(SAVE_DIR, 'prototypes.pkl')}")
