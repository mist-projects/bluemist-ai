# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.3
# Email: dew@bluemist-ai.one
# Created: Jul 17, 2023
# Last modified: Aug 18, 2023

from transformers import pipeline
import pandas as pd
from task_models import TaskModels

# Instantiate the TaskModels class
task_models = TaskModels()


# def deploy_model(task, model, context, question=None):
#     evaluate_models(task, context, question, override_models=model, limit=1)


def perform_task(task_name, context, question=None, override_models=None, limit=5, evaluate_models=True):
    """
        **Performs the task on the given dataset, evaluate the models and returns comparison metrics**

        task : str, default=None
            Supported tasks can be retrieved from the TaskModels class using the get_all_tasks method.
        context : str
            Text or information used by the model to perform specific NLP tasks.
        question : str, default=None
            Specific query or question provided as input to the model for question-answering tasks. The model uses this question to find the relevant answer within the provided context.
        override_models : str or list, default=None
            Provide additional models not part of the pre-configured list
        limit : int, default=5
            Limit the number of models to be compared. Default is 5.
    """

    # Check if the given task name is valid and supported by the available tasks.
    all_tasks = task_models.get_all_tasks()
    print(all_tasks)

    if task_name not in all_tasks:
        raise ValueError(f"Task '{task_name}' is not a valid task.")

    # Create an empty DataFrame to store and consolidate the results from different models
    results_df = pd.DataFrame()

    models = []
    if isinstance(override_models, str):
        if task_models.is_model_supported_by_task(override_models, task_name):
            models.append(override_models)
    elif isinstance(override_models, list):
        filtered_models = [model for model in override_models if
                           task_models.is_model_supported_by_task(model, task_name)]
        if filtered_models:
            models.extend(filtered_models)

    models.extend(task_models.get_models_for_task(task_name, limit))
    num_of_models = len(models)

    if not evaluate_models:
        limit = 1

    if limit is not None and 0 < limit <= num_of_models:
        models = models[:limit]

    if task_models.get_context_input_type(task_name) == "text":

        # Iterate through all available models for the task
        for model in models:
            print(f"Model: {model}")

            # Create the pipeline for the specific task
            nlp = pipeline(task=task_name, model=model)

            if task_models.is_question_supported(task_name):
                result = nlp(context=context, question=question)
            else:
                result = nlp(context)

            results_df = consolidate_results(result, results_df, model)

    return results_df


def consolidate_results(result, results_df, model):
    # Initialize an empty DataFrame
    new_rows_df = pd.DataFrame()

    if isinstance(result, dict):
        new_rows_df = pd.DataFrame([result])
    elif isinstance(result, list):
        new_rows_df = pd.DataFrame(result)

    new_rows_df.insert(0, 'model', model)

    # Store the results in the DataFrame
    results_df = pd.concat([results_df, new_rows_df], ignore_index=True)

    return results_df
