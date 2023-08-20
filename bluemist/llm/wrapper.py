# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.3
# Email: dew@bluemist-ai.one
# Created: Jul 17, 2023
# Last modified: Aug 20, 2023

from transformers import pipeline
import pandas as pd
from task_models import TaskModels

# Instantiate the TaskModels class
task_models = TaskModels()


def perform_task(task_name, input, question=None, override_models=None, limit=5, evaluate_models=True):
    """
        **Performs the task on the given dataset, evaluate the models and returns comparison metrics**

        task_name : str, default=None
            Supported tasks can be retrieved from the TaskModels class using the get_all_tasks method.
        input : str
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

    results_df = process_models(task_name, models, results_df, input, question)
    return results_df


def process_models(task_name, models, results_df, input, question=None):
    """
    Process multiple models with given inputs and consolidate results.

    Args:
        task_name : str, default=None
            Supported tasks can be retrieved from the TaskModels class using the get_all_tasks method.
        models : list
            A list of model names to be processed.
        results_df : pd.DataFrame
            The initial results DataFrame.
        context : str, default=None
            Text or information used by the model to perform specific NLP tasks.
        image : str, default=None
            The image input for image-based tasks. Defaults to None.
        question : str, default=None
            Specific query or question provided as input to the model for question-answering tasks. The model uses this question to find the relevant answer within the provided context.

    Returns:
        results_df : pd.DataFrame
            The updated results DataFrame containing consolidated results from all models.
    """

    for model in models:
        print(f"Model: {model}")

        nlp = pipeline(task=task_name, model=model)
        input_args = {}

        if task_models.is_question_supported(task_name):
            input_args["question"] = question

        if task_name == "question-answering":
            input_args["context"] = input
            result = nlp(input_args)
        elif task_name == "document-question-answering":
            input_args["image"] = input
            result = nlp(input_args)
        elif task_name == "summarization":
            input_args["min_length"] = 30
            input_args["max_length"] = 130
            input_args["do_sample"] = False
            result = nlp(input, **input_args)

        results_df = consolidate_results(result, results_df, model)
    return results_df


def consolidate_results(result, results_df, model):
    """
    Consolidates the given result into the results DataFrame.

    This function takes a result, which can be a dictionary or a list of dictionaries,
    and appends it to the provided results DataFrame. The 'model' argument is used to
    associate the result with a specific model.

    Args:
        result : (dict or list):
            The result to be consolidated into the DataFrame.
        results_df : (pd.DataFrame)
            The DataFrame to which the result will be added.
        model : str
            The model name associated with the result.

    Returns:
        results_df : pd.DataFrame
            The updated results DataFrame with the new consolidated result.
    """

    # Initialize an empty DataFrame
    new_rows_df = pd.DataFrame()

    # Some models return the result as dict while other may return list.
    if isinstance(result, dict):
        new_rows_df = pd.DataFrame([result])
    elif isinstance(result, list):
        new_rows_df = pd.DataFrame(result)

    new_rows_df.insert(0, 'model', model)

    # Store the results in the DataFrame
    results_df = pd.concat([results_df, new_rows_df], ignore_index=True)
    return results_df
