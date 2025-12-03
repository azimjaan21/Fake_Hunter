"""
Script to create a balanced dataset of fake images by sampling
400 images from each of 5 different source folders.
It copies a total of 2000 images into a specified destination folder.

1- StyleGAN2 generated images
2- StyleGAN3 generated images
3- Diffusion GAN generated images
4- Stable Diffusion generated images
5- Stable Diffusion generated images

"""

import os
import random
import shutil
from pathlib import Path

# --- Configuration ---
SOURCE_DIRS = [
    r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\artifact\diffusion_gan\diff\ffhq-data\Diffusion-StyleGAN2-ADA-FFHQ",
    r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\fake\run01",
    r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\artifact\stylegan2\ffhq-part1\ffhq-1024x1024\stylegan2-config-f-psi-0.5\000000",
    r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\artifact\stable_diffusion\stable-face\Female",
    r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\artifact\stable_diffusion\stable-face\Male",
]
DEST_DIR = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\metadata\fake"
IMAGES_PER_SOURCE = 400 # 5 sources * 400 images = 2000 total images
IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')

def copy_balanced_sample():
    # 1. Ensure destination directory exists and is empty before starting
    dest_path = Path(DEST_DIR)
    if dest_path.exists() and any(dest_path.iterdir()):
        print(f"⚠️ Destination directory {DEST_DIR} is not empty. Please clear it first!")
        return
    dest_path.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    
    # 2. Iterate through each source folder and sample 400 images
    for source_dir in SOURCE_DIRS:
        source_path = Path(source_dir)
        print(f"\nProcessing: {source_path.name}")
        
        # Get all image files in the current source directory
        all_files = [f for f in source_path.iterdir() if f.name.lower().endswith(IMG_EXTENSIONS)]
        
        if not all_files:
            print(f"❌ Warning: No images found in {source_dir}. Skipping.")
            continue

        num_to_sample = min(len(all_files), IMAGES_PER_SOURCE)
        
        # Select a random sample
        files_to_copy = random.sample(all_files, num_to_sample)
        
        # Copy the selected files to the destination
        copied_count = 0
        for file_path in files_to_copy:
            try:
                # Use the original filename to avoid conflicts, though random sampling makes it unlikely
                shutil.copy2(file_path, dest_path / file_path.name)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {file_path.name}: {e}")

        total_copied += copied_count
        print(f"✅ Copied {copied_count} files from this source.")

    print(f"\n--- Total Fake Images Copied: {total_copied} ---")
    print(f"Next: Run generate_meta_data.py on your new metadata folder!")

if __name__ == "__main__":
    copy_balanced_sample()