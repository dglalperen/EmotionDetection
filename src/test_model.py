import os
import numpy as np
from keras.models import load_model
from preprocessing import load_data
from keras.utils import to_categorical

def test_model(model_path, test_data_dir):
    # Load the trained model
    model = load_model(model_path)

    # Load test data
    test_images, test_labels = load_data(test_data_dir)
    
    print(f"Number of test images: {len(test_images)}")  # Debugging line

    if len(test_images) == 0:
        print("No images found in the test dataset. Please check the path and dataset.")
        return

    test_images = test_images.reshape(test_images.shape[0], 48, 48, 1)
    num_classes = 7  # Update if the number of classes is different
    test_labels = to_categorical(test_labels, num_classes)

    # Emotion label mapping
    emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

    # Test the model on each image in the test dataset
    correct_predictions = 0
    for i in range(len(test_images)):
        img = test_images[i]
        true_label = np.argmax(test_labels[i])
        img = np.expand_dims(img, axis=0)

        # Predict the emotion
        prediction = model.predict(img)
        predicted_label = np.argmax(prediction)

        # Print the results
        print(f"Image {i+1}: True Label - {emotion_labels[true_label]}, Predicted Label - {emotion_labels[predicted_label]}")
        if predicted_label == true_label:
            correct_predictions += 1

    # Calculate and print the accuracy
    accuracy = correct_predictions / len(test_images)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    
    model_path = 'emotion_model_optimized.h5'
    test_model(model_path, "../data/test")
