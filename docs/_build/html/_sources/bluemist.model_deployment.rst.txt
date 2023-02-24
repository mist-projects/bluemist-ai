Model Deployment
================

Bluemist provides integration with FastAPI to deploy the ML model as RESTful API. This can be simply achieved by calling ``deploy_model`` as described :ref:`here <bluemist.regression:regression>`

|

.. warning::
    ``deploy_model`` should only be executed after ``train_test_evaluate`` has been completed successfully

Examples
---------

.. raw:: html
    :file: ../code_samples/quickstarts/model_deployment/regression_model_deployment.html

|

To test the API, open the browser and navigate to `<http://localhost:8000/docs>`_

|

.. image:: ../code_samples/quickstarts/model_deployment/regression_model_deployment_fastapi_1.png

|

.. image:: ../code_samples/quickstarts/model_deployment/regression_model_deployment_fastapi_2.png
