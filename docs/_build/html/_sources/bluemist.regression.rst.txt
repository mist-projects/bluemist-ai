Regression
==========

.. automodule:: bluemist.regression.regressor
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: get_regressor_class, import_mlflow, initialize_mlflow, measure_execution_time


GPU Acceleration
----------------

- If your system is equipped with an NVIDIA GPU and you want to utilize GPU acceleration for model training, you can
  install the RAPIDS cuML package by executing the following command:

``pip install cudf-cu11 cuml-cu11 --extra-index-url=https://pypi.nvidia.com``

|

- The `cuml` package provides GPU-accelerated implementations of various machine learning algorithms. By installing this
  package, you can leverage the power of your GPU to significantly improve the speed and performance of your machine
  learning workflows.

- For the list of supported algorithms, please
  refer https://docs.rapids.ai/api/cuml/stable/api/#regression-and-classification

- Acceleration extensions can ben enabled by passing the parameter ``enable_acceleration_extensions=True`` during
  the ``initialize`` phase.

.. raw:: html
    :file: ../code_samples/quickstarts/regression/regression_gpu_acceleration_nvidia.html


CPU Acceleration
----------------

- If your system is equipped with an Intel CPU and you want to utilize CPU acceleration for model training, you can
  install the Intel® Extension for Scikit-learn package by executing the following command:

``pip install -U scikit-learn-intelex``

|

- The `scikit-learn-intelex` package provides CPU-accelerated implementations of Scikit-learn algorithms. By installing
  this package, you can leverage the power of your Intel CPU to significantly improve the speed and performance of your
  machine learning workflows.

- For the list of supported algorithms, please refer to
  the [Intel® Extension for Scikit-learn documentation](https://intel.github.io/scikit-learn-intelex/algorithms.html#on-cpu).

- Acceleration extensions can ben enabled by passing the parameter ``enable_acceleration_extensions=True`` during
  the ``initialize`` phase.

.. raw:: html
    :file: ../code_samples/quickstarts/regression/regression_cpu_acceleration_intel.html