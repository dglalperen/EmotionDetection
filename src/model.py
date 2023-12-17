from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf

def build_model(num_classes, hp_num_units, hp_dropout_rate):
    model = Sequential()

    # First Convolutional Block
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation(tf.nn.leaky_relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional Block
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(tf.nn.leaky_relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Convolutional Block
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(tf.nn.leaky_relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(hp_num_units, activation=tf.nn.leaky_relu))
    model.add(Dropout(hp_dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    return model
