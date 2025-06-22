# ♻ E-Waste Image Classifier

This project uses deep learning to classify electronic waste into various categories such as Battery, Mobile, PCB, and more. The goal is to assist in better waste management and recycling using image-based classification.

---

## 📌 Problem Statement

The rapid rise in electronic waste (e-waste) poses a significant environmental threat. Automating the classification of e-waste items can support recycling centers in sorting waste efficiently.

---

## 🎯 Objective

- Classify e-waste images into one of 10 predefined categories
- Achieve high classification accuracy using transfer learning
- Develop a reusable, scalable deep learning solution

---

## 🗂 Dataset Description

The dataset consists of labeled images across 10 e-waste categories:

- Battery  
- Mobile  
- Mouse  
- Keyboard  
- PCB (Printed Circuit Board)  
- Printer  
- Microwave  
- Washing Machine  
- Player (audio/video)  
- Television  

### Folder structure:
modified-dataset/
├── train/
├── val/
└── test/

yaml
Copy
Edit

Each folder contains subfolders named after the class labels, filled with relevant images.

---

## 🧠 Model Architecture

| Layer | Description |
|-------|-------------|
| Base Model | EfficientNetV2B0 (pretrained on ImageNet) |
| Global Average Pooling | Reduces dimensions |
| Dense (128 units) | ReLU activation |
| Dropout (0.3) | Prevents overfitting |
| Dense (10 units) | Softmax output |

---

## 📈 Results

| Metric              | Value   |
|---------------------|---------|
| Training Accuracy   | ~99.4%  |
| Validation Accuracy | ~97.6%  |
| Validation Loss     | ~0.12   |

---

## 📊 Visualizations

- Accuracy vs. Epochs  
- Loss vs. Epochs  
- Confusion Matrix

> All plots are generated automatically during training and evaluation.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Run the classifier
bash
Copy
Edit
python e_waste_classifier.py
3. Output
Trained model saved as e_waste_classifier.h5

Accuracy/loss plots

Confusion matrix

Sample predictions displayed

📦 Files Included
File	Description
e_waste_classifier.py	Main Python script (model + evaluation)
e_waste_classifier.h5	Trained model file
requirements.txt	Dependencies list
modified-dataset/	(Optional) Dataset (train/val/test)

⚙ Dependencies
These libraries are required (also in requirements.txt):

TensorFlow

NumPy

Matplotlib

Seaborn

scikit-learn

Pillow

✨ Author
Shrushti Barhate
📧 shrushtibarhate344@gmail.com
