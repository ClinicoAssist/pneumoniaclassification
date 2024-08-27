# Pneumonia Detection from Chest X-ray Images using Convolutional Neural Networks

This project implements a Convolutional Neural Network (CNN) to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**. The model is trained on a dataset containing X-ray images labeled accordingly. The project includes data preprocessing, model training, and evaluation with visualization of the results.

## Project Overview

The goal of this project is to develop a deep learning model that can accurately detect pneumonia from grayscale chest X-ray images. The CNN architecture consists of several convolutional layers with Batch Normalization, MaxPooling, and Dropout layers to prevent overfitting. The model's performance is enhanced by using data augmentation techniques during training.

### Key Features

- **Data Preprocessing**: 
  - Images are loaded from folders, converted to grayscale, resized, and normalized.
  - The dataset is split into training, validation, and testing sets.
  
- **Model Architecture**:
  - The CNN model includes multiple convolutional layers with ReLU activation, followed by MaxPooling and Dropout layers.
  - Batch Normalization is applied to stabilize and speed up the training.
  - The model concludes with a fully connected dense layer before the final classification layer.

- **Data Augmentation**:
  - The model utilizes the `ImageDataGenerator` class from Keras to apply data augmentation techniques such as rotation, shifting, shearing, zooming, and horizontal flipping.

- **Model Training**:
  - The model is trained using the Adam optimizer with sparse categorical cross-entropy loss.
  - Early stopping and learning rate reduction callbacks are used to optimize the training process.

- **Evaluation and Visualization**:
  - The training history (loss and accuracy) is plotted to visualize the model's learning over epochs.
  - A confusion matrix and classification report provide detailed insights into the model's performance.
  - The ROC curve is plotted to assess the model's capability to distinguish between the two classes.
  - Sample predictions are visualized alongside the true labels for qualitative evaluation.

- **Model Saving**:
  - The trained model is saved to an HDF5 file for future use.

## How to Use

1. **Dataset Preparation**: Ensure that your dataset is organized into separate folders for training, validation, and testing. Each of these folders should contain subfolders named after the classes (e.g., "NORMAL" and "PNEUMONIA").

2. **Run the Code**: 
   - Set the `base_path` variable in the `main` function to the path where your dataset is stored.
   - Execute the script. The model will be trained and evaluated, and the results will be visualized.

3. **Results**: 
   - The model's training history, confusion matrix, sample predictions, and ROC curve will be displayed.
   - The trained model will be saved as `pneumonia_model_improved.h5`.

## Requirements

- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

You can install the required libraries using pip:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn

