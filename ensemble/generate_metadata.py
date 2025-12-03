# generate_meta_data.py

import os
import sys
import csv
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Avoid PIL broken file errors

# ------------------------------------------------------------
# FIX PYTHON PATH so we can import discriminators properly
# ------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Now import
from discriminators import get_all_discriminators
# ------------------------------------------------------------

# --- CONFIGURATION ---
DATASET_PATH = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\metadata"
OUTPUT_CSV = "meta_train.csv"
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')


# ------------------------------------------------------------
# Auto-detect model type (GAN-D vs classifier)
# ------------------------------------------------------------
def get_model_type(model_name):
    name = model_name.lower()
    if any(k in name for k in ['stylegan', 'gan_d', 'diffusion', 'progan']):
        return 'gan_d'
    return 'classifier'


# ------------------------------------------------------------
# Process one image across all discriminators
# ------------------------------------------------------------
def process_image(models, img_path):
    try:
        img = Image.open(img_path).convert("RGB")

        logits = []
        types = []

        # Run all models
        for m in models:
            logit = m.score(img)   # YOU ALREADY HAVE THIS
            logits.append(logit)
            types.append(get_model_type(m.name))

        # Convert logits to probabilities
        probs = []
        for logit, m_type in zip(logits, types):

            # 1. Flip polarity for GAN-D (Real=positive)
            score = -logit if m_type == 'gan_d' else logit

            # 2. Stabilize (clip extreme scores)
            score = np.clip(score, -20, 20)

            # Sigmoid → probability of FAKE
            p_fake = 1.0 / (1.0 + np.exp(-score))
            probs.append(p_fake)

        return probs

    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")
        return None


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    # Load discriminators once
    print("[INFO] Loading models...")
    models = get_all_discriminators()
    model_names = [m.name for m in models]

    print(f"[INFO] Loaded {len(models)} models:")
    for n in model_names:
        print("   →", n)

    headers = model_names + ["target"]

    # Create CSV
    with open(OUTPUT_CSV, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # --- Process REAL ---
        real_dir = Path(DATASET_PATH) / "real"
        if real_dir.is_dir():
            print(f"\n[REAL] Processing: {real_dir}")
            image_files = [p for p in real_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]

            for path in tqdm(image_files, ncols=80):
                probs = process_image(models, path)
                if probs:
                    writer.writerow(probs + [0])  # Real = 0

        # --- Process FAKE ---
        fake_dir = Path(DATASET_PATH) / "fake"
        if fake_dir.is_dir():
            print(f"\n[FAKE] Processing: {fake_dir}")
            image_files = [p for p in fake_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]

            for path in tqdm(image_files, ncols=80):
                probs = process_image(models, path)
                if probs:
                    writer.writerow(probs + [1])  # Fake = 1

    print("\n------------------------------------------------------------")
    print(f"[DONE] Metadata saved → {OUTPUT_CSV}")
    print("Now run:   python train_meta.py")
    print("------------------------------------------------------------")


if __name__ == "__main__":
    main()
