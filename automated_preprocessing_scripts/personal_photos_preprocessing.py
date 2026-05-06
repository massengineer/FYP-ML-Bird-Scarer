import cv2
import os

# Setup Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(BASE_DIR, "raw_backgrounds")
processed_dir = os.path.join(BASE_DIR, "final_background_dataset")

# --- STEP 2: BATCH PREPROCESS (OPENCV) ---
os.makedirs(processed_dir, exist_ok=True)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

search_path = raw_dir
if os.path.exists(os.path.join(raw_dir, "data")):
    search_path = os.path.join(raw_dir, "data")

image_files = [
    f for f in os.listdir(search_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Normalizing and Resizing {len(image_files)} images...")
for img_name in image_files:
    path = os.path.join(search_path, img_name)
    img = cv2.imread(path)
    if img is None:
        continue

    # Normalisation
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Resize
    img = cv2.resize(img, (640, 640))
    cv2.imwrite(os.path.join(processed_dir, img_name), img)

print(f"Success! Your balanced dataset is ready in: {processed_dir}")
