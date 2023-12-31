import os
from tensorflow import keras
import keras_tuner as kt
from model import build_model
from preprocessing import load_data
from keras.utils import to_categorical
from keras.optimizers.legacy import Adam

# Load preprocessed data
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

train_images, train_labels = load_data(train_dir)
test_images, test_labels = load_data(test_dir)
num_classes = 7
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

def model_builder(hp):
    # Expanded range of hyperparameters
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    hp_num_units = hp.Int('num_units', min_value=64, max_value=256, step=64)

    # Build and compile the model
    model = build_model(num_classes, hp_num_units, hp_dropout_rate)
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize the tuner
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='hyperparam_opt')

# Early stopping callback
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Execute the search
tuner.search(train_images, train_labels, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"The optimal learning rate is {best_hps.get('learning_rate')}.")
print(f"The optimal number of units is {best_hps.get('num_units')}.")
print(f"The optimal dropout rate is {best_hps.get('dropout_rate')}.")

# Build the model with the optimal hyperparameters
model = tuner.hypermodel.build(best_hps)

# Train the model
model.fit(train_images, train_labels, epochs=50, validation_split=0.2)

# Save the model
model.save('emotion_model_optimized.h5')
