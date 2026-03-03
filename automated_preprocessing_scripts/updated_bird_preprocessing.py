import fiftyone as fo
import fiftyone.zoo as foz
import cv2
import os
import shutil

# 1. Setup Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(BASE_DIR, "raw_downloads")
processed_dir = os.path.join(BASE_DIR, "final_bird_dataset")

# --- CLEANUP FIFTYONE DATABASE ---
# This loop clears out ALL existing datasets so you start with 0MB of DB usage
print("Cleaning up FiftyOne database...")
for ds_name in fo.list_datasets():
    fo.delete_dataset(ds_name)

# --- CLEANUP LOCAL FOLDERS ---
if os.path.exists(raw_dir): shutil.rmtree(raw_dir)
if os.path.exists(processed_dir): shutil.rmtree(processed_dir)

# --- STEP 1: DOWNLOAD & MERGE (BALANCED) ---
# We use a master dataset to hold our final 390 images
main_dataset = fo.Dataset(name="final_balanced_birds")

bird_species = ["Duck", "Raven", "Sparrow"]

for species in bird_species:
    print(f"--- Downloading {species} ---")
    # Using a unique temp name for each download helps prevent memory leaks
    temp_ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=[species],
        max_samples=100,
        only_matching=True,
        shuffle=True,
        dataset_name=f"temp_{species}"
    )
    main_dataset.add_samples(temp_ds)
    fo.delete_dataset(f"temp_{species}") # Clear the temp data from RAM/DB immediately

print("--- Downloading Backgrounds ---")
bg_ds = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=[], # No labels needed for backgrounds
    classes=["Tree", "Building"],
    max_samples=90,
    shuffle=True,
    dataset_name="temp_bg"
)
main_dataset.add_samples(bg_ds)
fo.delete_dataset("temp_bg")

# Export
print(f"Exporting to: {raw_dir}")
main_dataset.export(
    export_dir=raw_dir, 
    dataset_type=fo.types.ImageDirectory,
    overwrite=True
)

# --- STEP 2: BATCH PREPROCESS (OPENCV) ---
os.makedirs(processed_dir, exist_ok=True)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

search_path = raw_dir
if os.path.exists(os.path.join(raw_dir, "data")):
    search_path = os.path.join(raw_dir, "data")

image_files = [f for f in os.listdir(search_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Normalizing and Resizing {len(image_files)} images...")
for img_name in image_files:
    path = os.path.join(search_path, img_name)
    img = cv2.imread(path)
    if img is None: continue

    # Normalization
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Resize
    img = cv2.resize(img, (640, 640))
    cv2.imwrite(os.path.join(processed_dir, img_name), img)

print(f"Success! Your balanced dataset is ready in: {processed_dir}")