import numpy as np
import os
from model import build_model
from preprocessing import load_data, train_test_split
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from keras.optimizers.legacy import Adam

# Define paths
print("Current Working Directory:", os.getcwd())
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Load training and testing data
train_images, train_labels = load_data(train_dir)
test_images, test_labels = load_data(test_dir)

# Reshape and one-hot encode labels
train_images = train_images.reshape(train_images.shape[0], 48, 48, 1)
test_images = test_images.reshape(test_images.shape[0], 48, 48, 1)
num_classes = 7
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Split training data for validation
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Calculate class weights for imbalanced data
train_labels_single = np.argmax(train_labels, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_single), y=train_labels_single)
class_weights_dict = dict(enumerate(class_weights))

# Data Augmentation
augmentor = ImageDataGenerator(
    rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.15, horizontal_flip=True,
    fill_mode="nearest")

# Build the model with optimized hyperparameters
model = build_model(num_classes=7, hp_num_units=192, hp_dropout_rate=0.4)

# Compile the model with optimized learning rate
adam_optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Model Checkpoint and Early Stopping
checkpoint = ModelCheckpoint('emotion_model_best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

# Train the model
batch_size = 64
epochs = 20
history = model.fit(
    augmentor.flow(train_images, train_labels, batch_size=batch_size),
    steps_per_epoch=len(train_images) // batch_size,
    validation_data=(val_images, val_labels),
    epochs=epochs,
    callbacks=[checkpoint, early_stopping],
    class_weight=class_weights_dict
)

# Save the final model
model.save('emotion_model_optimized.h5')

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc * 100:.2f}%")