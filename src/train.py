import numpy as np
import os
from model import build_model
from preprocessing.preprocessing import load_data, train_test_split
from keras.utils import to_categorical

# Define paths
print("Current Working Directory:", os.getcwd())
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Load training and testing data
train_images, train_labels = load_data(train_dir)
test_images, test_labels = load_data(test_dir)

# One-hot encode labels
num_classes = 7  # Number of emotions
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Split training data for validation
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Build the model
model = build_model(num_classes)

# Train the model with optimized parameters
batch_size = 64
epochs = 20   
history = model.fit(
    train_images, train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(val_images, val_labels)
)

# Save the model in the new Keras format
model.save('emotion_model.h5')


# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
