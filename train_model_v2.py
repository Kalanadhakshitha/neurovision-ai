import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

MAIN_DIR = "dataset"
SUB_DIRS = ["Training", "Testing"]
CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

IMG_SIZE = 128 
data = []

print("Reading data from both folders started...")

# Reading data (Looping through Training & Testing)
for sub_dir in SUB_DIRS:
    for category in CATEGORIES:
        # path is like: dataset/Training/glioma
        path = os.path.join(MAIN_DIR, sub_dir, category) 
        
        try:
            class_num = CATEGORIES.index(category) 

            if not os.path.exists(path):
                print(f"Skipping: {path} (Folder cannot be found)")
                continue

            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    data.append([new_array, class_num])
                except Exception as e:
                    pass
        except Exception as e:
            print(f"Error accessing {path}")

print(f"Total number of images: {len(data)}")

# Data Preparation 
import random
random.shuffle(data)

X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

X = X / 255.0

# Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multi-class CNN Model 
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    
    
    layers.Dense(4, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the Model
print("Training පටන් ගත්තා... (Multi-class)")
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

# Results
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save Model
model.save('brain_tumor_multiclass.h5')
print("Model saved as 'brain_tumor_multiclass.h5'!")

# Graph
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()