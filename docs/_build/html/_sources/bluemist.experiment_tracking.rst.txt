Experiment Tracking
===================

Bluemist provides integration with MLflow to track, experiment and evaluate machine learning models. This can be simply achieved by setting the params ``experiment_name`` and ``run_name`` while training the model using ``train_test_evaluate`` as described :ref:`here <bluemist.regression:regression>`

|

.. note::
    Is is important to execute ``!mlflow ui`` after ``train_test_evaluate`` has been completed successfully

Examples
---------

.. raw:: html
    :file: ../code_samples/quickstarts/experiment_tracking/regression_experiment_tracking.html

|

To track the experiments, open the browser and navigate to `<http://127.0.0.1:5000>`_

|

.. image:: ../code_samples/quickstarts/experiment_tracking/regression_experiment_tracking_ui.png