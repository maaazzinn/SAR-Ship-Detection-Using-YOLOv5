import subprocess
import os

# Path to your trained weights
MODEL_PATH = os.path.abspath("sar_output/ship_detector_run/weights/best.pt")

# Path to test image or folder
TEST_IMAGE_PATH = r"C:\Users\mazin\Downloads\ship detection\datasets\SSDD_coco\images\000017.jpg"

# Ensure you're in yolov5 directory
os.chdir("yolov5")

# Run detection
subprocess.run([
    "python", "detect.py",
    "--weights", MODEL_PATH,
    "--img", "640",
    "--conf", "0.25",
    "--source", TEST_IMAGE_PATH,
    "--project", "../sar_output",
    "--name", "ship_predictions",
    "--exist-ok"
], check=True)

print("✅ Inference done — check 'sar_output/ship_predictions'")
