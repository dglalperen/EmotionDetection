import numpy as np
from model import build_model
from preprocessing.preprocessing import load_data, train_test_split
from keras.utils import to_categorical
from keras.optimizers.legacy import Adam
import os

# Load preprocessed data
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

train_images, train_labels = load_data(train_dir)
test_images, test_labels = load_data(test_dir)
num_classes = 7
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Define hyperparameters to test
batch_sizes = [32, 64]
epochs = [10, 20]
learning_rates = [0.001, 0.0001]

best_accuracy = 0
best_params = {}

for batch_size in batch_sizes:
    for epoch in epochs:
        for learning_rate in learning_rates:
            print(f"Training with batch size={batch_size}, epochs={epoch}, learning rate={learning_rate}")

            # Build and compile the model
            model = build_model(num_classes)
            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(learning_rate=learning_rate),
                          metrics=['accuracy'])

            # Train the model
            history = model.fit(
                train_images, train_labels,
                batch_size=batch_size,
                epochs=epoch,
                validation_data=(test_images, test_labels)
            )

            # Evaluate the model
            val_loss, val_acc = model.evaluate(test_images, test_labels)

            # Update best accuracy and parameters
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_params = {'batch_size': batch_size, 'epochs': epoch, 'learning_rate': learning_rate}

print(f"Best Accuracy: {best_accuracy}")
print(f"Best Parameters: {best_params}")
