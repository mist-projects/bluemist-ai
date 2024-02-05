"""
Performs model training, testing, evaluations and deployment
"""
import os

# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.4
# Email: dew@bluemist-ai.one
# Created:  Dec 29, 2023
# Last modified: Feb 04, 2024

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Nadam, Adamax
from tensorflow.keras.initializers import get
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_test_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        task=None,
        epochs=1000,
        batch_size=None,
        kernel_initializer=None,
        loss=None,
        metrics=None,
        optimizer_name=None,
        learning_rate=None,
        optimizer_params=None,
        layer_configs=None,
        early_stopping_kwargs=None,
        model_checkpoint_kwargs=None,
        plot_graphs=True
):
    # Select optimizer dynamically based on user input
    if optimizer_name is not None:
        optimizer_name = optimizer_name.lower()

        # Initialize learning rate if None
        if learning_rate is None:
            learning_rate = 0.001

        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate, **optimizer_params) \
                if optimizer_params is not None and isinstance(optimizer_params, dict) else Adam(
                learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, **optimizer_params) \
                if optimizer_params is not None and isinstance(optimizer_params, dict) else SGD(
                learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate, **optimizer_params) \
                if optimizer_params is not None and isinstance(optimizer_params, dict) else RMSprop(
                learning_rate=learning_rate)
        elif optimizer_name == 'adagrad':
            optimizer = RMSprop(learning_rate=learning_rate, **optimizer_params) \
                if optimizer_params is not None and isinstance(optimizer_params, dict) else RMSprop(
                learning_rate=learning_rate)
        elif optimizer_name == 'adadelta':
            optimizer = Adadelta(learning_rate=learning_rate, **optimizer_params) \
                if optimizer_params is not None and isinstance(optimizer_params, dict) else Adadelta(
                learning_rate=learning_rate)
        elif optimizer_name == 'nadam':
            optimizer = Nadam(learning_rate=learning_rate, **optimizer_params) \
                if optimizer_params is not None and isinstance(optimizer_params, dict) else Nadam(
                learning_rate=learning_rate)
        elif optimizer_name == 'adamax':
            optimizer = Adamax(learning_rate=learning_rate, **optimizer_params) \
                if optimizer_params is not None and isinstance(optimizer_params, dict) else Adamax(
                learning_rate=learning_rate)
        else:
            raise ValueError(f"Invalid optimizer: {optimizer_name}")
    else:
        # Use a default optimizer if not specified
        optimizer = Adam(learning_rate=learning_rate)

    # Define input layer
    input_shape = X_train.shape[1]
    input_layer = Input(shape=(input_shape,))
    model_output_layer = input_layer

    if task == 'regression' and layer_configs is None:
        layer_configs = [
            {'units': 1024, 'activation': 'relu'},
            {'units': 512, 'activation': 'relu'},
            {'units': 256, 'activation': 'relu'},
            {'units': 128, 'activation': 'relu'},
            {'units': 96, 'activation': 'relu'},
            {'units': 1, 'activation': 'linear'}
        ]
    elif task == 'classification' and layer_configs is None:
        layer_configs = [
            {'units': 1024, 'activation': 'relu'},
            {'units': 512, 'activation': 'relu'},
            {'units': 256, 'activation': 'relu'},
            {'units': 128, 'activation': 'relu'},
            {'units': 96, 'activation': 'relu'},
            {'units': len(np.unique(y_train)), 'activation': 'softmax'}
        ]

    dense_layers = []
    for config in layer_configs:
        dense_layer = Dense(**config)(model_output_layer)
        dense_layers.append(dense_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=dense_layers)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.summary()
    callbacks_list = []

    # Define early stopping callback
    if early_stopping_kwargs is None:
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks_list.append(early_stopping_callback)
    else:
        early_stopping_callback = EarlyStopping(**early_stopping_kwargs)
        callbacks_list.append(early_stopping_callback)

    BLUEMIST_PATH = os.environ["BLUEMIST_PATH"]
    MODELS_PATH = BLUEMIST_PATH + '/' + 'artifacts' + '/' + 'models'

    # Define model checkpoint callback
    if model_checkpoint_kwargs is None:
        model_checkpoint_callback = ModelCheckpoint(MODELS_PATH + 'best_model.h5', monitor='val_loss', save_best_only=True)
        callbacks_list.append(model_checkpoint_callback)
    else:
        model_checkpoint_kwargs['filepath'] = os.path.join(MODELS_PATH, model_checkpoint_kwargs['filepath'])
        model_checkpoint_callback = ModelCheckpoint(**model_checkpoint_kwargs)
        callbacks_list.append(model_checkpoint_callback)

    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=2, callbacks=callbacks_list)

    if plot_graphs:
        if task == 'regression':
            plot_loss(history)
        # elif task == 'classification':
        #     plot_accuracy(history, task='classification')
        #     plot_metrics(history, metric='precision')
        #     #plot_confusion_matrix(y_true_classification, y_pred_classification, classes=class_labels)

    # # Evaluate on the test data
    # test_metrics = model.evaluate(X_test, y_test)
    return model


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
