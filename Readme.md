# Plant Disease Detection Using Deep Learning

## Overview

This project leverages deep learning and transfer learning to automatically detect plant diseases from leaf images. It uses a convolutional neural network (CNN) based on MobileNetV2, trained on the [PlantVillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage) dataset, which contains thousands of labeled images across 38 plant disease classes. The project includes a Streamlit web application for easy image-based inference.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Web Application](#web-application)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Dataset](#dataset)


---

## Features

- **Transfer Learning**: Uses MobileNetV2 pretrained on ImageNet for feature extraction.
- **Data Augmentation**: Improves generalization using rotation, shift, flip, zoom, and shear.
- **Multi-Class Classification**: Supports 38 plant disease classes.
- **Performance Metrics**: Reports accuracy, precision, recall, and confusion matrix.
- **Interactive Web App**: Upload leaf images and get instant predictions.

---

## Project Structure

```
.
├── app.py                          # Streamlit web app for inference
├── class_labels.json               # Mapping of class indices to names
├── experiment.ipynb                # Jupyter notebook for training and evaluation
├── plant_disease_detection_model.h5# Trained Keras model
├── Readme.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── data/
│   └── PlantVillage/
│       ├── train/                  # Training images organized by class
│       └── val/                    # Validation images organized by class
```

---

## Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/plant-disease-detection.git
   cd plant-disease-detection
   ```

2. **Create and activate a virtual environment**
   ```sh
   python -m venv venv
   venv\Scripts\activate   # On Windows
   ```

3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Download the PlantVillage dataset**
   - Place the dataset in `data/PlantVillage/train` and `data/PlantVillage/val` as per the structure above.

---

## Usage

### Training

- Open `experiment.ipynb` in Jupyter or VS Code.
- Run all cells to train the model, visualize metrics, and save the trained model and class labels.

### Evaluation

- The notebook computes accuracy, precision, recall, and displays a confusion matrix for validation data.
- Training history plots are generated for accuracy and loss.

### Web Application

- Start the Streamlit app:
  ```sh
  streamlit run app.py
  ```
- Upload a leaf image (`.jpg`, `.jpeg`, `.png`) and click "Analyze Image" to get the predicted disease class and confidence score.

---

## Model Architecture

- **Base Model**: MobileNetV2 (pretrained, feature extractor)
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense (256 units, ReLU) + Dropout (0.3)
  - Dense (128 units, ReLU) + Dropout (0.2)
  - Output: Dense (softmax, 38 classes)

- **Training Details**:
  - Optimizer: Adam
  - Loss: Categorical Crossentropy
  - Metrics: Accuracy, Precision, Recall
  - Early stopping and learning rate reduction callbacks

---

## Results

- **Validation Accuracy**: ~92%
- **Precision & Recall**: High across most classes
- **Confusion Matrix**: Visualized in the notebook

Sample training history and evaluation plots are available in `experiment.ipynb`.

---

## Dataset

- **Source**: [PlantVillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)
- **Classes**: 38 plant diseases and healthy categories
- **Images**: ~43,000 training, ~10,000 validation

