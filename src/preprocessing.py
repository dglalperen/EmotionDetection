import os
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(directory):
    print("Loading data from", directory)
    images = []
    labels = []
    img_width, img_height = 48, 48

    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    for idx, emotion in enumerate(emotions):
        emotion_dir = os.path.join(directory, emotion.lower())
        print(f"Loading {emotion} images from: {emotion_dir}")

        if not os.path.isdir(emotion_dir):
            print(f"Directory not found: {emotion_dir}")
            continue

        for img_file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_file)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue
                img = cv2.resize(img, (img_width, img_height))
                images.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    if not images:
        print("No images found. Check the dataset.")
        return np.array([]), np.array([])

    images = np.array(images, dtype='float32') / 255.0  # Normalize the images
    labels = np.array(labels, dtype='int')

    return images, labels



def preprocess_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', '..', 'data')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    num_classes = 7  # Number of emotion classes

    # Load and preprocess the data
    train_images, train_labels = load_data(train_dir)
    test_images, test_labels = load_data(test_dir)

    # One-hot encode the labels
    train_labels = to_categorical(train_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    # Split the training data for validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

if __name__ == "__main__":
    train_images, train_labels, val_images, val_labels, test_images, test_labels = preprocess_data()
    print("Data loaded and preprocessed.")
