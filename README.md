# Cat vs Dog Classification Using CNN

This project uses a Convolutional Neural Network (CNN) to classify images of cats and dogs. The dataset consists of labeled images divided into training and validation sets. After training the model for 20 epochs, an accuracy of **80%** was achieved on the validation dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [Training](#training)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

---

## Overview
This project demonstrates a binary classification task using a simple CNN architecture to distinguish between images of cats and dogs. The model is trained on an augmented dataset to improve generalization.

---

## Dataset
- **Source**: [Kaggle Cat vs Dog Dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)
- **Structure**:
  - `train/`: Contains subfolders for `cat` and `dog` images used for training.
  - `validation/`: Contains subfolders for `cat` and `dog` images used for validation.

---

## Model Architecture
The CNN architecture consists of:
1. **Convolutional Layers**:
   - Extract spatial features using `Conv2D` with ReLU activation.
   - Downsample using `MaxPooling2D`.
2. **Fully Connected Layers**:
   - Flatten the feature maps.
   - Use a Dense layer with 512 neurons and ReLU activation.
   - Dropout layer to reduce overfitting.
3. **Output Layer**:
   - Dense layer with 1 neuron and sigmoid activation for binary classification.

### Model Summary
```
Layer (type)                 Output Shape              Param #
=================================================================
Conv2D (32 filters, 3x3)     (None, 148, 148, 32)      896
MaxPooling2D                 (None, 74, 74, 32)        0
Conv2D (64 filters, 3x3)     (None, 72, 72, 64)        18496
MaxPooling2D                 (None, 36, 36, 64)        0
Conv2D (128 filters, 3x3)    (None, 34, 34, 128)       73856
MaxPooling2D                 (None, 17, 17, 128)       0
Flatten                      (None, 37056)             0
Dense (512 neurons)          (None, 512)               18949184
Dropout                      (None, 512)               0
Dense (1 neuron)             (None, 1)                 513
=================================================================
Total params: 19,134,945
Trainable params: 19,134,945
Non-trainable params: 0
```

---

## Data Augmentation
To improve model generalization, data augmentation was applied to the training set using `ImageDataGenerator`:
- Rescaling pixel values.
- Random rotations (up to 40 degrees).
- Width and height shifts (up to 20%).
- Random shear and zoom.
- Horizontal flipping.

---

## Training
- **Optimizer**: Adam (learning rate = 0.001).
- **Loss Function**: Binary Crossentropy.
- **Batch Size**: 32.
- **Epochs**: 20.
- **Steps Per Epoch**: Calculated based on training samples and batch size.
- **Validation Steps**: Calculated based on validation samples and batch size.

---

## Results
- **Validation Accuracy**: **80%** after 20 epochs.
- **Training Accuracy**: Converged steadily, indicating the model effectively learned from the dataset.

### Accuracy Curve
The following plot demonstrates the training and validation accuracy over epochs:

![image](https://github.com/user-attachments/assets/1f3c3ec1-7c53-441f-84be-e632d202b8a3)


---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/DeMoN-7/Cat-vs-dog-using-CNN.git
   ```
2. Install required dependencies:
   ```bash
   pip install tensorflow matplotlib
   ```
3. Download the dataset from the link above and organize it as follows:
   ```
   dataset/
   ├── train/
   │   ├── cat/
   │   ├── dog/
   ├── validation/
       ├── cat/
       ├── dog/
   ```

---

## Usage
1. Train the model:
   ```python
   python main.py
   ```
2. Evaluate the model on the validation set:
   ```python
   python evaluate.py
   ```

---

## References
- TensorFlow Documentation: https://www.tensorflow.org/
- Kaggle Dataset: https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset
