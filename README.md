# 🐱🐶 Cat vs Dog Image Classifier using SVM

This project implements an image classification model using **Support Vector Machines (SVM)** to distinguish between cats and dogs. It uses grayscale image data from the [PetImages dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765), resizes them, and flattens the images before classification.

---

## 🧠 Model Overview

- **Model Used:** Support Vector Classifier (SVC) with RBF kernel
- **Input Size:** 64 x 64 grayscale images
- **Features:** Flattened pixel intensities
- **Preprocessing:** Resizing, normalization, and standard scaling

---

## 🗂️ Dataset Structure


task3/
├── PetImages/
│ ├── Cat/
│ │ ├── cat1.jpg
│ │ └── ...
│ └── Dog/
│ ├── dog1.jpg
│ └── ...
├── svm_cat_dog_classifier.py
└── README.md



- Only `.jpg` images are expected inside the `Cat/` and `Dog/` folders.
- Make sure images are not corrupt; the script skips unreadable ones automatically.

---

## ⚙️ Configuration

Inside the script:

dataset_path = r'task3\PetImages'
IMG_SIZE = 64
n_per_class = 2000  # Number of images per class to load (set to None to use all)
You can change n_per_class to use a specific number of images from each class.

🛠️ How to Run
1. Install Requirements

pip install numpy opencv-python matplotlib scikit-learn tqdm
2. Run the Script


python svm_cat_dog_classifier.py
🔍 What the Script Does
Loads grayscale images from both Cat and Dog folders.

Resizes images to 64x64.

Normalizes pixel values and flattens each image to a 1D array.

Splits the data into training and validation sets.

Scales features using StandardScaler.

Trains an SVM with an RBF kernel.

Evaluates validation accuracy.

Randomly visualizes 5 validation predictions along with probability scores.

📊 Output Example
✅ Loaded 4000 images.

🎯 Validation Accuracy: ~80–90% depending on dataset sample.

🖼️ Visualization showing predicted vs. true labels with confidence.

📁 Output
No files are saved, but predictions are printed and visualized using matplotlib.

📚 References
Dataset: Microsoft Dogs vs Cats Dataset

Scikit-learn SVM: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

🏷️ Tags
SVM Image Classification Cats vs Dogs Computer Vision OpenCV Python Machine Learning



