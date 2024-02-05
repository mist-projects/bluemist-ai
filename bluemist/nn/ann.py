"""
Performs model training, testing, evaluations and deployment
"""
import os

# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.4
# Email: dew@bluemist-ai.one
# Created:  Dec 29, 2023
# Last modified: Feb 05, 2024

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

from bluemist.utils.metrics import metric_scorer

BLUEMIST_PATH = os.environ["BLUEMIST_PATH"]
MODELS_PATH = BLUEMIST_PATH + '/' + 'artifacts' + '/' + 'models'


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
    """
    Train, test, and evaluate a neural network model.

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training input data.
    X_test : pandas.DataFrame
        Test input data.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    task : str, optional
        The task type ('regression' or 'classification').
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Batch size for training.
    kernel_initializer : str, optional
        Initializer for kernel weights.
    loss : str, optional
        Loss function to use during training.
    metrics : list, optional
        List of metrics to monitor during training.
    optimizer_name : str, optional
        Name of the optimizer to use.
    learning_rate : float, optional
        Learning rate for the optimizer.
    optimizer_params : dict, optional
        Additional parameters for the optimizer.
    layer_configs : list, optional
        List of dictionaries specifying layer configurations.
    early_stopping_kwargs : dict, optional
        Additional parameters for EarlyStopping callback.
    model_checkpoint_kwargs : dict, optional
        Additional parameters for ModelCheckpoint callback.
    plot_graphs : bool, optional
        Whether to plot graphs (e.g., loss) during training.

    Returns:
    --------
    history : tensorflow.python.keras.callbacks.History
        History object containing training metrics.
    model : tensorflow.keras.models.Model
        Trained neural network model.
    """

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

    # Add hidden layers
    for config in layer_configs[:-1]:  # Exclude the last config for the output layer
        model_output_layer = Dense(**config)(model_output_layer)

    # Output layer for regression
    output_layer = Dense(**layer_configs[-1])(model_output_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.summary()
    callbacks_list = []

    # Define early stopping callback
    if early_stopping_kwargs is not None:
        early_stopping_callback = EarlyStopping(**early_stopping_kwargs)
        callbacks_list.append(early_stopping_callback)

    # Define model checkpoint callback
    if model_checkpoint_kwargs is not None:
        model_checkpoint_kwargs['filepath'] = os.path.join(MODELS_PATH, model_checkpoint_kwargs['filepath'])
        model_checkpoint_callback = ModelCheckpoint(**model_checkpoint_kwargs)
        callbacks_list.append(model_checkpoint_callback)

    history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, verbose=2)
    model.summary()
    model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    scorer = metric_scorer(y_test, y_pred, 'default')

    estimator_stats_df = scorer.calculate_metrics()
    print(estimator_stats_df)

    return history, model
