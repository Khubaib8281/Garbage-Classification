# Garbage Classification using Deep Learning

This project implements an end-to-end deep learning pipeline for garbage image classification. The entire workflow, from data loading to model training, evaluation, saving, loading, and testing on unseen images, is contained in a single Jupyter notebook.

> The model is trained on a real-world garbage dataset and achieves ~90% validation accuracy across 12 classes.

## Classes

The classifier predicts one of the following categories:
- battery
- biological
- brown-glass
- cardboard
- clothes
- green-glass
- metal
- paper
- plastic
- shoes
- trash
- white-glass
  
## Dataset

Name: Garbage Classification Dataset

Source: mostafaabla/garbage-classification (Kaggle)

Data Type: Labeled RGB images

Structure: Class-wise folders

The dataset contains natural variations in lighting, background, and object orientation, making it suitable for practical waste classification tasks.

## Notebook Overview

The notebook covers the full pipeline:

- Data Loading
- Reads images directly from class directories
- Splits data into training and validation sets
- Applies preprocessing and normalization
- Data Augmentation
- Rotation, flipping, and scaling
- Helps reduce overfitting

## Model Creation

CNN-based architecture

Designed for multi-class image classification

Easily extendable to transfer learning models

Training

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Trained for multiple epochs with validation tracking

## Evaluation

Validation accuracy reaches ~90%

Performance monitored using accuracy and loss curves

Model Saving and Loading

Trained model is saved to disk

Reloaded for inference without retraining

Testing on Custom Images

Supports prediction on unseen test images

Outputs predicted class with confidence score

## Results

Validation Accuracy: ~90%

Strong performance on common garbage types such as paper, plastic, and metal

Some confusion between similar glass categories, which is expected given visual overlap

## How to Use

Clone the repository

Download the dataset from Kaggle:

mostafaabla/garbage-classification


Place the dataset in the expected directory path (as referenced in the notebook)

### Open the notebook:

garbage_classification.ipynb


Run all cells sequentially

No separate training or inference scripts are required.

### Requirements

Python 3.x

TensorFlow / PyTorch

NumPy

Matplotlib

OpenCV or PIL

Jupyter Notebook

(Exact versions are listed or can be inferred from the notebook.)

### Limitations

Model performance may degrade on heavily cluttered or low-quality images

Glass subclasses remain the hardest to distinguish

Not optimized for real-time or edge deployment

### Future Work

Add per-class precision and recall

Introduce a confusion matrix

Experiment with pretrained backbones

Export to ONNX or TensorRT

Deploy as a lightweight web demo

### Acknowledgements

Dataset by Mostafa Abla

Open-source deep learning ecosystem
