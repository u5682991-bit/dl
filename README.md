# X-ray Pneumonia Detection

A binary classification project using deep learning to identify pneumonia/opacity in chest X-rays. Built with PyTorch and three different CNN architectures.

## Models
Trained and compared:
- VGG16
- ResNet50
- MobileNetV2

## Data Structure
\\\
dataset/
├── train/
│   ├── normal/
│   └── opacity/
├── val/
│   ├── normal/
│   └── opacity/
└── test/
    ├── normal/
    └── opacity/
\\\

## Tech Stack
- Python 3.11
- PyTorch 2.2.1 + Torchvision
- scikit-learn, Pandas, NumPy
- Matplotlib, OpenCV

## Hardware
- CPU: Intel Core i9-13980HX
- GPU: NVIDIA RTX 4080 (Laptop)

## Quick Start
1. Open \u5682991.ipynb\
2. Run cells in order
3. Check the comparison tables and Grad-CAM visualizations at the end

## Output
Trained models saved as:
- \est_model_vgg16.pth\
- \est_model_resnet.pth\
- \est_model_mobilenet.pth\

Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
