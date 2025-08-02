import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ========== CONFIG ==========
dataset_path = r'task3\PetImages'  # Use raw string or forward slashes
IMG_SIZE = 64
n_per_class = 2000  # Change to None for all images
# ============================

# Function to load images from Cat and Dog folders
def load_images_from_folders(root_dir, n_per_class=None):
    X, y = [], []
    classes = {'Cat': 0, 'Dog': 1}  # Match exact folder names

    for class_name, label in classes.items():
        folder = os.path.join(root_dir, class_name)
        files = sorted(os.listdir(folder))[:n_per_class] if n_per_class else os.listdir(folder)

        for file in tqdm(files, desc=f"Loading {class_name}"):
            img_path = os.path.join(folder, file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype(np.float32) / 255.0
                img_flat = img.flatten()
                X.append(img_flat)
                y.append(label)
            except Exception as e:
                continue  # Skip unreadable or corrupt images

    X = np.array(X)
    y = np.array(y)
    return X, y

# Load images
X, y = load_images_from_folders(dataset_path, n_per_class=n_per_class)
print(f"\n✅ Loaded {len(X)} images.")
print("Class distribution:", np.bincount(y))

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train SVM
print("\n🚀 Training SVM...")
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"\n🎯 Validation Accuracy: {accuracy * 100:.2f}%")

# Visualize predictions
def visualize_predictions(num_samples=5):
    print(f"\n🔍 Showing {num_samples} validation predictions...\n")
    indices = np.random.choice(len(X_val), num_samples, replace=False)

    for idx in indices:
        sample = X_val[idx].reshape(1, -1)
        prob = model.predict_proba(sample)[0][1]
        pred_label = "Dog" if prob > 0.5 else "Cat"
        true_label = "Dog" if y_val[idx] == 1 else "Cat"

        # Try to display the corresponding image (approximate back-mapping)
        original_folder = 'Dog' if y_val[idx] == 1 else 'Cat'
        folder_path = os.path.join(dataset_path, original_folder)
        image_files = sorted(os.listdir(folder_path))
        if idx < len(image_files):
            img_path = os.path.join(folder_path, image_files[idx])
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(3, 3))
            plt.imshow(img_rgb)
            plt.title(f"Predicted: {pred_label} ({prob:.2f})\nTrue: {true_label}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

# Visualize 5 predictions
visualize_predictions(num_samples=5)
