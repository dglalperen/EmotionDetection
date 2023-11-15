import os
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Determine the absolute path to the directory containing preprocessing.py
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to base_dir
data_dir = os.path.join(base_dir, '..', '..', 'data')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Image parameters
img_width, img_height = 48, 48

def load_data(directory):
    images = []
    labels = []

    for idx, emotion in enumerate(os.listdir(directory)):
        # Ensure directory
        if not os.path.isdir(os.path.join(directory, emotion)):
            continue

        for img_file in os.listdir(os.path.join(directory, emotion)):
            img_path = os.path.join(directory, emotion, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_width, img_height))  # Ensuring size consistency

            images.append(img)
            labels.append(idx)

    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='float32')

    # Normalizing images to a range from 0 to 1
    images = images / 255.0

    return images, labels

# Loading training and testing data
train_images, train_labels = load_data(train_dir)
test_images, test_labels = load_data(test_dir)

# One-hot encode labels
num_classes = 7  # As specified for the emotions
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Split training data for validation
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

print("Data loaded and preprocessed.")
