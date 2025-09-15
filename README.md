# Pneumonia Detection from Chest X-Ray Images Using Deep Learning

**Author:** Matthew Martin  
**Project:** Mini-Project 3  

---

## Introduction

### Problem Statement
Pneumonia is a serious respiratory condition that can be life-threatening if not diagnosed early. Automated detection from chest X-ray images using deep learning can support radiologists, reduce diagnostic errors, and improve patient outcomes.

### Goal
Develop and compare deep learning models to classify chest X-ray images into two categories:

- Normal  
- Pneumonia  

---

## Dataset

**Chest X-Ray Images (Pneumonia)** dataset from Kaggle:

- **Total images:** 5,863  
- **Train:** 5,216 images  
- **Test:** 624 images  
- **Classes:** Normal, Pneumonia  

**Class Distribution**

| Dataset    | Normal | Pneumonia |
|------------|--------|-----------|
| Train      | 1,341  | 3,895     |
| Validation | 8      | 8         |
| Test       | 234    | 390       |

**Notes:**

- Images are in JPEG format with three channels (RGB).  
- Average resolution: ~2090×1858 pixels.  
- Images were resized to 224×224 pixels for model input.  

---

## Exploratory Data Analysis (EDA)

- **Class Distribution:** Slight imbalance in training data, with Pneumonia > Normal.  
- **Pixel Intensities:** Pneumonia images show slightly darker pixels due to lung consolidations.  
- **Duplicates & Corrupt Images:** Checked and removed duplicates; no corrupt images detected.  
- **Visualization:** Random samples from each class verified correct labeling.  

---

## Data Preprocessing

- Resized all images to 224×224 pixels.  
- Normalized pixel values to [0,1].  
- Applied data augmentation during training: horizontal flips, rotations, zooms.  
- Computed class weights to address imbalance:  
  - Normal: 1.95  
  - Pneumonia: 0.67  

---

## Model Architectures

### Custom Convolutional Neural Network (CNN)

- Three convolutional blocks with Conv2D → BatchNormalization → MaxPooling.  
- GlobalAveragePooling → Dense(128) → Dropout → Sigmoid output.  
- Parameters: ~111k  
- Test Accuracy: ~86%  

### Transfer Learning Model (VGG16)

- Pretrained VGG16 base (frozen layers) → GlobalAveragePooling → Dense(256) → Dropout → Sigmoid output.  
- Parameters: ~14.8M (trainable: ~131k)  
- Test Accuracy: ~86%, with strong recall for Pneumonia (0.92).  

**Evaluation Metrics (Transfer Model)**

| Metric     | Normal | Pneumonia | Overall |
|------------|--------|-----------|---------|
| Precision  | 0.86   | 0.87      | 0.86    |
| Recall     | 0.76   | 0.92      | 0.86    |
| F1-Score   | 0.81   | 0.89      | 0.86    |

---

## Training Details

- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Epochs:** 15 (Custom CNN), 10 (Transfer Learning)  
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  
- **Visualization:** Training/validation accuracy & loss, confusion matrix, ROC curve  

---

## Potential Improvements

- Expand the validation set for more reliable hyperparameter tuning.  
- Experiment with other pretrained models such as ResNet or EfficientNet.  
- Implement more sophisticated data augmentation techniques.  
- Fine-tune VGG16 layers to further improve performance.  

---

## Conclusion

- Successfully detected pneumonia using deep learning.  
- Transfer learning (VGG16) outperformed the custom CNN in accuracy and robustness.  
- The model shows potential for supporting clinical decision-making in healthcare imaging.  

---


