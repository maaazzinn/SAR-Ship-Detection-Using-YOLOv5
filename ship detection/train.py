import os
import json
import shutil
from PIL import Image
import subprocess

# === Step 1: Paths ===
MAIN_DIR = 'datasets\SSDD_coco'  # Change this if needed
IMG_DIR = os.path.join(MAIN_DIR, 'images')
ANN_DIR = os.path.join(MAIN_DIR, 'annotations')
YOLO_ROOT = 'SAR_Ship_YOLO'

# === Step 2: Setup Directory Structure ===
IMG_TRAIN = os.path.join(YOLO_ROOT, 'images', 'train')
LBL_TRAIN = os.path.join(YOLO_ROOT, 'labels', 'train')

os.makedirs(IMG_TRAIN, exist_ok=True)
os.makedirs(LBL_TRAIN, exist_ok=True)

# === Step 3: Convert LabelMe JSON → YOLOv5 TXT ===
def polygon_to_bbox(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return x_min, y_min, x_max, y_max

def convert_annotations():
    print("🔁 Converting annotations...")
    for filename in os.listdir(ANN_DIR):
        if filename.endswith('.json'):
            json_path = os.path.join(ANN_DIR, filename)
            with open(json_path, 'r') as f:
                data = json.load(f)

            image_filename = data['imagePath']
            image_path = os.path.join(IMG_DIR, image_filename)
            label_path = os.path.join(LBL_TRAIN, filename.replace('.json', '.txt'))

            # Copy image to YOLO image folder
            shutil.copy(image_path, os.path.join(IMG_TRAIN, image_filename))

            img = Image.open(image_path)
            img_width, img_height = img.size

            with open(label_path, 'w') as out:
                for shape in data['shapes']:
                    x_min, y_min, x_max, y_max = polygon_to_bbox(shape['points'])
                    x_center = ((x_min + x_max) / 2) / img_width
                    y_center = ((y_min + y_max) / 2) / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height
                    out.write(f"0 {x_center} {y_center} {width} {height}\n")

    print("✅ Conversion complete!")

# === Step 4: Create sar.yaml ===
def create_yaml():
    yaml_content = f"""
train: {os.path.abspath(IMG_TRAIN)}
val: {os.path.abspath(IMG_TRAIN)}  # using same for val if no split

nc: 1
names: ['ship']
"""
    with open("sar.yaml", "w") as f:
        f.write(yaml_content.strip())
    print("📄 sar.yaml created.")

# === Step 5: Run Training ===
def run_training():
    print("🚀 Starting YOLOv5 training...")

    if not os.path.isdir("yolov5"):
        try:
            subprocess.run([
                "git", "clone", "https://github.com/ultralytics/yolov5"
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Git clone failed: {e}")
            return

    os.chdir("yolov5")

    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

    subprocess.run([
        "python", "train.py",
        "--img", "640",
        "--batch", "8",
        "--epochs", "20",
        "--data", "../sar.yaml",
        "--weights", "yolov5s.pt",
        "--project", "../sar_output",
        "--name", "ship_detector_run",
        "--exist-ok"
    ], check=True)

# === Run All ===
if __name__ == "__main__":
    convert_annotations()
    create_yaml()
    run_training()
