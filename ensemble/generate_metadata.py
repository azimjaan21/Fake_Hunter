# generate_meta_data.py
import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm  # pip install tqdm

# Import your actual model loader
# Adjust 'discriminators' to match your actual folder/file name
from discriminators import get_all_discriminators
from ensemble.ensemble_models import DeepHunterEnsemble, sigmoid

# --- CONFIGURATION ---
DATASET_PATH = "meta_dataset"  # Folder containing 'real' and 'fake' subfolders
OUTPUT_CSV = "meta_train.csv"
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')

def get_model_type(model_name):
    """Auto-detect model type for polarity check."""
    name = model_name.lower()
    if any(k in name for k in ['stylegan', 'diffusion', 'gan_d', 'progan']):
        return 'gan_d'
    return 'classifier'

def process_image(models, img_path):
    """Runs all models on one image and returns a list of probabilities."""
    try:
        # Open and convert image
        img = Image.open(img_path).convert("RGB")
        
        # 1. Get Logits
        logits = []
        model_types = []
        for m in models:
            logits.append(m.score(img))
            model_types.append(get_model_type(m.name))
        
        # 2. Convert Logits to Probabilities (Manual Step)
        # We do this here to give the Meta-Learner clean [0-1] inputs
        probs = []
        for logit, m_type in zip(logits, model_types):
            if m_type == 'gan_d':
                score = -logit # Polarity Flip
            else:
                score = logit
            
            # Sigmoid Calibration
            score = np.clip(score, -20, 20)
            p_fake = 1 / (1 + np.exp(-score))
            probs.append(p_fake)
            
        return probs

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def main():
    # 1. Load Models Once
    print("Loading models... (this may take a moment)")
    models = get_all_discriminators()
    model_names = [m.name for m in models]
    print(f"Loaded {len(models)} models: {model_names}")

    # 2. Prepare CSV File
    # Columns: [Model1_Prob, Model2_Prob, ..., True_Label]
    headers = model_names + ["target"]
    
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # 3. Process Real Images (Target = 0)
        real_folder = os.path.join(DATASET_PATH, 'real')
        if os.path.exists(real_folder):
            print(f"Processing REAL images in {real_folder}...")
            for fname in tqdm(os.listdir(real_folder)):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    path = os.path.join(real_folder, fname)
                    probs = process_image(models, path)
                    if probs:
                        writer.writerow(probs + [0]) # 0 = Real

        # 4. Process Fake Images (Target = 1)
        fake_folder = os.path.join(DATASET_PATH, 'fake')
        if os.path.exists(fake_folder):
            print(f"Processing FAKE images in {fake_folder}...")
            for fname in tqdm(os.listdir(fake_folder)):
                if fname.lower().endswith(IMG_EXTENSIONS):
                    path = os.path.join(fake_folder, fname)
                    probs = process_image(models, path)
                    if probs:
                        writer.writerow(probs + [1]) # 1 = Fake

    print(f"\nDone! Data saved to {OUTPUT_CSV}")
    print("Next Step: Run 'train_meta.py' to create your meta_model.pkl")

if __name__ == "__main__":
    main()