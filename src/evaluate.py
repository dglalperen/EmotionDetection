from keras.models import load_model
import numpy as np
import itertools
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from preprocessing import load_data
from keras.utils import to_categorical
import os

emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
num_classes = len(emotion_labels)

print("Evaluating model...")
# Load the trained model
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
test_dir = os.path.join(data_dir, 'test')

model = load_model('./emotion_model_optimized.h5')
test_images, test_labels = load_data(test_dir)
test_labels = to_categorical(test_labels, num_classes)

# Get the model's predictions
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Classification Report
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

# ROC Curves and AUC (adapted for multi-class)
y_bin = label_binarize(true_classes, classes=np.arange(num_classes))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], predictions[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve (optional, if needed)
# for i in range(num_classes):
#     precision, recall, _ = precision_recall_curve(y_bin[:, i], predictions[:, i])
#     plt.plot(recall, precision, label=f'Class {i}')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc="lower left")
# plt.show()
