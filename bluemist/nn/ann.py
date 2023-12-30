"""
Performs model training, testing, evaluations and deployment
"""

# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.4
# Email: dew@bluemist-ai.one
# Created:  Dec 29, 2023
# Last modified: Dec 30, 2023

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


def train_test_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        task=None,
        epochs=None,
        batch_size=None,
        kernel_initializer=None):

    # Define input layer
    input_shape = X_train.shape[1]
    representation_size = X_train.shape[0]
    factor = 2
    base_units = 256

    # Calculate the number of layers and units dynamically
    num_layers = int(np.ceil(np.log2(representation_size))) + 1
    layer_units = [int(np.ceil(factor * input_shape / (2 ** i)) + base_units) for i in range(1, num_layers + 1)]

    input_layer = Input(shape=(input_shape,))
    model_output_layer = input_layer
    for units in layer_units:
        model_output_layer = Dense(units=units, activation='relu')(model_output_layer)

    # Build the model based on the task
    if task == 'regression':
        if kernel_initializer is None:
            kernel_initializer = glorot_normal()
        model_output_layer = Dense(units=1, activation='linear', kernel_initializer=kernel_initializer)(model_output_layer)
        loss = 'mean_squared_error'
        metrics = [mean_squared_error]
    elif task == 'classification':
        if kernel_initializer is None:
            kernel_initializer = 'he_normal'
        # Adjust activation based on the number of classes (e.g., 'softmax' for multi-class)
        model_output_layer = Dense(units=1, activation='sigmoid', kernel_initializer=kernel_initializer)(model_output_layer)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        raise ValueError("Invalid task. Supported tasks: 'regression' or 'classification'.")

    # Create the model
    model = Model(inputs=input_layer, outputs=model_output_layer)
    model.compile(optimizer=Adam(), loss=loss, metrics=metrics)

    model.summary()

    # Train the model with validation split
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    # Evaluate on the test data
    test_metrics = model.evaluate(X_test, y_test)

    return model, test_metrics


def plot_loss(history):
    # Plot training loss
    plt.plot(history.history['loss'], label='Train Loss')

    # Check if validation loss is available in the history
    if 'val_loss' in history.history:
        # Plot validation loss
        plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()