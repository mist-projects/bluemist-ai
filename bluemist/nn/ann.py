"""
Performs model training, testing, evaluations and deployment
"""

# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.4
# Email: dew@bluemist-ai.one
# Created:  Dec 29, 2023
# Last modified: Jan 22, 2024

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Nadam, Adamax
from tensorflow.keras.initializers import get
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def train_test_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        task=None,
        epochs=None,
        batch_size=None,
        kernel_initializer=None,
        num_layers=None,
        num_neurons=None,
        activation=None,
        loss=None,
        metrics=None,
        optimizer_name=None,
        learning_rate=None,
        optimizer_params=None,
        plot_graphs=True
):
    # Define input layer
    input_shape = X_train.shape[1]

    # Use user-provided or default values for the number of layers and neurons
    num_layers = num_layers or 2
    num_neurons = num_neurons or 128

    # Define input layer
    input_layer = Input(shape=(input_shape,))
    model_output_layer = input_layer

    # Build the model
    for _ in range(num_layers):
        model_output_layer = Dense(units=num_neurons, activation='relu')(model_output_layer)

    # Build the model based on the task
    if task == 'regression':
        kernel_initializer = get(kernel_initializer)
        model_output_layer = Dense(units=1, activation='linear', kernel_initializer=kernel_initializer)(
            model_output_layer)
        loss = 'mean_squared_error'
        metrics = [mean_squared_error]
    elif task == 'classification':
        kernel_initializer = get(kernel_initializer)

        # Adjust activation based on the number of classes
        num_classes = 2  # Default for binary classification
        if y_train is not None:
            num_classes = len(np.unique(y_train))

        if num_classes == 2:
            # Binary classification
            activation = activation if activation is not None else 'sigmoid'
            loss = loss if loss is not None else 'binary_crossentropy'
        else:
            # Multi-class classification
            activation = activation if activation is not None else 'softmax'
            loss = loss if loss is not None else ('categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy')

            # Use 'sparse_categorical_crossentropy' if the labels are integers
            if isinstance(y_train[0], (int, np.integer)):
                loss = 'sparse_' + loss

            metrics = metrics if metrics is not None else ['accuracy']

        model_output_layer = Dense(units=num_classes, activation=activation, kernel_initializer=kernel_initializer)(model_output_layer)
    else:
        raise ValueError("Invalid task. Supported tasks: 'regression' or 'classification'.")

    # Select optimizer dynamically based on user input
    if optimizer_name is not None:
        optimizer_name = optimizer_name.lower()
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate, **optimizer_params)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, **optimizer_params)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate, **optimizer_params)
        elif optimizer_name == 'adagrad':
            optimizer = Adagrad(learning_rate=learning_rate, **optimizer_params)
        elif optimizer_name == 'adadelta':
            optimizer = Adadelta(learning_rate=learning_rate, **optimizer_params)
        elif optimizer_name == 'nadam':
            optimizer = Nadam(learning_rate=learning_rate, **optimizer_params)
        elif optimizer_name == 'adamax':
            optimizer = Adamax(learning_rate=learning_rate, **optimizer_params)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer_name}")
    else:
        # Use a default optimizer if not specified
        optimizer = Adam()

    # Create the model
    model = Model(inputs=input_layer, outputs=model_output_layer)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.summary()

    # Train the model with validation split
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    if plot_graphs:
        if task == 'regression':
            plot_loss(history)
            plot_accuracy(history, task='regression')
            plot_metrics(history, metric='mean_squared_error')
        elif task == 'classification':
            plot_accuracy(history, task='classification')
            plot_metrics(history, metric='precision')
            #plot_confusion_matrix(y_true_classification, y_pred_classification, classes=class_labels)

    # Evaluate on the test data
    test_metrics = model.evaluate(X_test, y_test)
    return model, test_metrics


def plot_loss(history):
    """
    Plot training and validation loss from the training history.

    Parameters:
    - history: Training history.
    """
    plt.plot(history.history['loss'], label='Train Loss')

    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_accuracy(history, task='classification'):
    """
    Plot training and validation accuracy from the training history.

    Parameters:
    - history: Training history.
    - task: Type of task ('classification' or 'regression').
    """
    metric = 'accuracy' if task == 'classification' else 'mean_squared_error'

    plt.plot(history.history[metric], label='Train ' + metric.capitalize())

    if 'val_' + metric in history.history:
        plt.plot(history.history['val_' + metric], label='Validation ' + metric.capitalize())

    plt.title('Training and Validation ' + metric.capitalize())
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()


def plot_metrics(history, metric='precision'):
    """
    Plot training and validation metrics from the training history.

    Parameters:
    - history: Training history.
    - metric: Metric to plot (default is 'precision').
    """
    plt.plot(history.history[metric], label='Train ' + metric.capitalize())

    if 'val_' + metric in history.history:
        plt.plot(history.history['val_' + metric], label='Validation ' + metric.capitalize())

    plt.title('Training and Validation ' + metric.capitalize())
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()


def plot_learning_rate(history):
    """
    Plot learning rate schedule from the training history.

    Parameters:
    - history: Training history.
    """
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.show()


def plot_confusion_matrix(y_true, y_pred, classes=None):
    """
    Plot confusion matrix for classification tasks.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - classes: Class labels (optional).
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()