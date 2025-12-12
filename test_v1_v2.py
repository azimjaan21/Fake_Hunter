import os
import cv2
import numpy as np
import pickle
from glob import glob
from tqdm import tqdm
from PIL import Image

from discriminators import get_all_discriminators
from ensemble.ensemble_models import DeepHunterEnsemble
from ensemble.embedder import EnsembleEmbedder
from ensemble.protonet import predict_with_prototypes

# ===============================================
# 1. USER SETTINGS
# ===============================================
IMAGE_DIR = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\Fake_Hunter\data\support\fake"  # folder with fake images

# label for all images in this folder
TRUE_LABEL = 1   # 1=fake, 0=real


# ===============================================
# 2. ENSEMBLE + FEWSHOT MODELS
# ===============================================

_ENSEMBLE = None
_DETECTORS = None
_EMBEDDER = None
_PROTOTYPES = None
_PROTO_CLASSES = None
_FAKE_CLASS_IDX = None


def _init_ensemble():
    """Lazy-load heavy discriminators once per process."""
    global _ENSEMBLE, _DETECTORS
    if _ENSEMBLE is not None:
        return

    _DETECTORS = get_all_discriminators()
    cfg = []
    for d in _DETECTORS:
        name = getattr(d, "name", "detector").lower()
        if "gan" in name:
            m_type = "gan_d"
        elif "vae" in name:
            m_type = "vae"
        else:
            m_type = "classifier"
        cfg.append({"name": name, "type": m_type, "weight": 1.0})

    _ENSEMBLE = DeepHunterEnsemble(cfg)


def _init_protonet():
    """Lazy-load ProtoNet components and cached prototypes."""
    global _EMBEDDER, _PROTOTYPES, _PROTO_CLASSES, _FAKE_CLASS_IDX
    if _EMBEDDER is not None and _PROTOTYPES is not None:
        return

    _init_ensemble()
    _EMBEDDER = EnsembleEmbedder(_DETECTORS)

    proto_path = os.path.join("fewshot_cache", "prototypes.pkl")
    if not os.path.exists(proto_path):
        raise FileNotFoundError(
            f"ProtoNet prototypes not found at {proto_path}. "
            "Run test_create_prototypes.py to generate them."
        )

    with open(proto_path, "rb") as f:
        data = pickle.load(f)

    _PROTOTYPES = data["prototypes"]
    _PROTO_CLASSES = data["classes"]
    fake_idxs = np.where(_PROTO_CLASSES == 1)[0]
    _FAKE_CLASS_IDX = int(fake_idxs[0]) if len(fake_idxs) else None


def predict_v1(img):
    """
    DeepHunter V1 – Ensemble fusion-based detection.
    Returns:
        0 -> real
        1 -> fake
    """
    _init_ensemble()
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    logits = [d.score(pil_img) for d in _DETECTORS]
    result = _ENSEMBLE.predict(logits)
    return 1 if result["final_prob"] >= 0.5 else 0


def predict_v2(img):
    """
    DeepHunter V2 – ProtoNet + Few-shot embedding-based detection.
    Returns:
        0 -> real
        1 -> fake
    """
    _init_protonet()
    if _FAKE_CLASS_IDX is None:
        raise RuntimeError("Prototypes do not include fake class (1).")

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    q = _EMBEDDER.extract_single(pil_img).reshape(1, -1)

    probs, _ = predict_with_prototypes(q, _PROTOTYPES, metric="cosine")
    prob_fake = float(probs[0][_FAKE_CLASS_IDX])
    return 1 if prob_fake >= 0.5 else 0


# ===============================================
# 3. ACCURACY EVALUATION FUNCTION
# ===============================================

def evaluate_model(predict_fn, image_paths, name="Model"):
    correct = 0
    total = len(image_paths)

    for img_path in tqdm(image_paths, desc=f"Evaluating {name}"):
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to load:", img_path)
            continue

        pred = predict_fn(img)
        if pred == TRUE_LABEL:
            correct += 1

    acc = (correct / total) * 100
    print(f"\n{name} Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc


# ===============================================
# 4. MAIN EXECUTION
# ===============================================
if __name__ == "__main__":
    image_paths = sorted(
        glob(os.path.join(IMAGE_DIR, "*.jpg")) +
        glob(os.path.join(IMAGE_DIR, "*.png")) +
        glob(os.path.join(IMAGE_DIR, "*.jpeg"))
    )

    print(f"Loaded {len(image_paths)} images from: {IMAGE_DIR}")

    # --- Evaluate V1 ---
    acc_v1 = evaluate_model(predict_v1, image_paths, name="DeepHunter V1")

    # --- Evaluate V2 ---
    acc_v2 = evaluate_model(predict_v2, image_paths, name="DeepHunter V2")

    print("\n================ FINAL SUMMARY ================")
    print(f"DeepHunter V1 (Ensemble Fusion) Accuracy: {acc_v1:.2f}%")
    print(f"DeepHunter V2 (ProtoNet Few-Shot) Accuracy: {acc_v2:.2f}%")
    print("================================================")
