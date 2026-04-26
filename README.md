# Pneumonia Detection Project

This project performs binary classification on chest X-ray images to detect Normal vs Pneumonia/Opacity.
The workflow is implemented in Jupyter Notebook with PyTorch-based transfer learning.

## Project Scope
- Task type: Binary image classification
- Labels: normal, opacity (or pneumonia in compatible datasets)
- Goal: High-recall medical screening support with interpretable visualization

## Models Used
- VGG16
- ResNet50
- MobileNetV2

## Dataset Layout
Expected folder structure:

- train/
	- normal/
	- opacity/
- val/
	- normal/
	- opacity/
- test/
	- normal/
	- opacity/

The notebook also includes class alias handling so different folder naming styles can still be mapped correctly.

## Environment
- Python 3.11
- PyTorch + Torchvision
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

## Hardware (Notebook/Laptop)
- CPU: Intel Core i9-13980HX
- GPU: NVIDIA GeForce RTX 4080 (Laptop)

## Main Notebook
- u5682991.ipynb

## Training and Evaluation Pipeline
1. Initialize random seeds and runtime configuration.
2. Resolve project path and validate dataset structure.
3. Build dataloaders with normalization and augmentation.
4. Train transfer-learning models (VGG16, ResNet50, MobileNetV2).
5. Evaluate with Accuracy, Precision, Recall, F1-Score, and AUC-ROC.
6. Compare models and generate final summary tables.
7. Visualize model attention using Grad-CAM (configured for MobileNetV2).

## Saved Artifacts
- best_model_vgg16.pth
- best_model_resnet.pth
- best_model_mobilenet.pth

## How To Run
1. Open u5682991.ipynb.
2. Run all cells from top to bottom in order.
3. Ensure training cells complete before Grad-CAM cells.
4. Review final comparison tables and heatmap visualizations.

## Notes
- If kernel is restarted, rerun prerequisite cells before downstream evaluation/visualization.
- For faster execution, keep GPU enabled in the selected notebook kernel.
