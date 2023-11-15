from keras.models import load_model
import numpy as np
import itertools
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from preprocessing.preprocessing import load_data
from keras.utils import to_categorical
import os
from keras.optimizers.legacy import Adam

emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}


print("Evaluating model...")
# Load the trained model
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This navigates up to the root directory
data_dir = os.path.join(base_dir, 'data')
test_dir = os.path.join(data_dir, 'test')

model = load_model('./emotion_model.h5')
test_images, test_labels = load_data(test_dir)
test_labels = to_categorical(test_labels, num_classes=7)

# Get the model's predictions
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Calculate the classification report
report = classification_report(true_classes, predicted_classes, target_names=emotion_labels.values(), zero_division=0)

print(report)

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plotting Confusion Matrix
plt.figure(figsize=(5, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(emotion_labels))
plt.xticks(tick_marks, emotion_labels.values(), rotation=45)
plt.yticks(tick_marks, emotion_labels.values())

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
