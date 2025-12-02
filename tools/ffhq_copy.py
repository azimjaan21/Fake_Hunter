import os
import random
import shutil

# --- Configuration ---
SOURCE_DIR = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\data\artifact\ffhq\ffhq\images"
DEST_DIR = r"C:\Users\dalab\Desktop\azimjaan21\DeepHUNTER\fakeface_generator\metadata\real"
NUM_IMAGES_TO_COPY = 2000

def copy_random_sample():
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    print(f"Amount: {NUM_IMAGES_TO_COPY}")

    # 1. Ensure destination directory exists
    os.makedirs(DEST_DIR, exist_ok=True)
    
    # 2. Get a list of all files in the source directory
    # Filter to only include common image file extensions
    all_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not all_files:
        print("\n❌ Error: No image files found in the source directory.")
        return

    # 3. Check if the source has enough files
    if len(all_files) < NUM_IMAGES_TO_COPY:
        print(f"\n⚠️ Warning: Source only has {len(all_files)} files. Copying all of them.")
        files_to_copy = all_files
    else:
        # 4. Select a random sample without replacement
        files_to_copy = random.sample(all_files, NUM_IMAGES_TO_COPY)

    # 5. Copy the selected files
    print("\nStarting copy process...")
    for filename in files_to_copy:
        source_path = os.path.join(SOURCE_DIR, filename)
        dest_path = os.path.join(DEST_DIR, filename)
        shutil.copy2(source_path, dest_path) # copy2 preserves metadata

    print("\n✅ Copying complete!")
    print(f"Total files copied to {DEST_DIR}: {len(files_to_copy)}")

if __name__ == "__main__":
    copy_random_sample()