# SAR Ship Detection Using YOLOv5 🚢📡

This repository presents the implementation of a YOLOv5-based object detection model for identifying ships in Synthetic Aperture Radar (SAR) images. The project includes dataset preparation, annotation conversion, model training, and evaluation using custom SAR data.

---

## 📑 Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

---

## 📖 Introduction

Synthetic Aperture Radar (SAR) imagery presents unique challenges for object detection tasks due to speckle noise and variable backscatter characteristics. This project leverages the **YOLOv5** architecture to detect ships in SAR imagery, using a tailored training pipeline for dataset preparation, conversion, and training automation.

---

## 📦 Dataset

- **Source**: SAR Ship Detection Dataset (SSDD)
- **Annotation format**: Originally in LabelMe JSON format
- **Preprocessing**:
  - Converted LabelMe polygon annotations to YOLO bounding box format.
  - Organized dataset into YOLOv5-compatible folder structure.
  - Split data into training and validation sets.

---

## ⚙️ Methodology

**Pipeline Overview:**

1. Convert JSON annotations to YOLO format.
2. Set up YOLO dataset directory structure.
3. Generate a training configuration `.yaml` file.
4. Train YOLOv5 using pretrained `yolov5s.pt` weights.
5. Evaluate the model using object detection metrics.

**Training Parameters:**

- Image size: `640`
- Batch size: `8`
- Epochs: `20`
- Pretrained weights: `yolov5s.pt`

---

## 💾 Installation

```bash
# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

1. **Prepare Dataset**  
   Place images and converted YOLO-format labels in the following structure:

   ```
   datasets/
     └── SSDD_YOLO/
         ├── images/
         │   ├── train/
         │   └── val/
         └── labels/
             ├── train/
             └── val/
   ```

2. **Run Training**

   ```bash
   python train.py --img 640 --batch 8 --epochs 20 --data SSDD.yaml --weights yolov5s.pt
   ```

---

## 📊 Results

| Metric         | Value |
|:---------------|:--------|
| **Precision**      | 0.968  |
| **Recall**         | 0.970  |
| **mAP@0.5**        | 0.991  |
| **mAP@0.5:0.95**   | 0.711  |

Additional visual outputs:
- Loss curves
- Confusion matrix
- F1, Precision-Recall, Confidence curves
- Label distribution heatmaps

---

## ✅ Conclusion

The YOLOv5 model achieved **near-perfect precision and recall**, with a high **mAP@0.5 of 0.991** on SAR ship detection, demonstrating that YOLOv5 is highly effective for real-time maritime surveillance tasks in complex SAR imagery environments.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- SSDD Dataset authors for providing the SAR data.
