# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.3
# Email: dew@bluemist-ai.one
# Created: Jul 17, 2023
# Last modified: Oct 25, 2023

import logging
import os
import pandas as pd
from logging import config

from transformers import pipeline
from bluemist.llm.task_models import TaskModels

BLUEMIST_PATH = os.environ["BLUEMIST_PATH"]

config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")

# Instantiate the TaskModels class
task_models = TaskModels()


def perform_task(task_name, input_data, question=None, min_length=30, max_length=130, do_sample=False,
                 override_models=None, limit=5, evaluate_models=True):
    """
        **Performs the task on the given dataset, evaluate the models and returns comparison metrics**

        task_name : str, default=None
            Supported tasks can be retrieved from the TaskModels class using the get_all_tasks method.
        input_data : str
            Text or information used by the model to perform specific NLP tasks.
        question : str, default=None
            Specific query or question provided as input to the model for question-answering tasks. The model uses this question to find the relevant answer within the provided context.
        min_length: number, default=30
            The minimum length of the generated summary. Defaults to 30. The summarization model ensures that the summary is at least this length.
        max_length : number, default=130
            The maximum length of the generated summary. Defaults to 130. The summarization model limits the summary to a maximum of this length.
        do_sample : boolean, default=False
            Whether to use sampling during summary generation. Defaults to False. When True, the model uses a sampling technique for token selection.
        override_models : str or list, default=None
            Provide additional models not part of the pre-configured list
        limit : int, default=5
            Limit the number of models to be compared. Default is 5.
        evaluate_models : boolean, default=True
            Determine if model comparison is requested. ``False`` will override `limit` as 1
    """

    # Check if the given task name is valid and supported by the available tasks.
    all_tasks = task_models.get_all_tasks()

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

    results_df = process_models(task_name, models, results_df, input_data, question, min_length, max_length, do_sample)
    return results_df


def process_models(task_name, models, results_df, input_data, question=None, min_length=30, max_length=130,
                   do_sample=False):
    """
    Process multiple models with given inputs and consolidate results.

    Args:
        task_name : str, default=None
            Supported tasks can be retrieved from the TaskModels class using the get_all_tasks method.
        models : list
            A list of model names to be processed.
        results_df : pd.DataFrame
            The initial results DataFrame.
        input_data : str, default=None
            Text or information used by the model to perform specific NLP tasks.
        question : str, default=None
            Specific query or question provided as input to the model for question-answering tasks. The model uses this question to find the relevant answer within the provided context.
        min_length: number, default=30
            The minimum length of the generated summary. Defaults to 30. The summarization model ensures that the summary is at least this length.
        max_length : number, default=130
            The maximum length of the generated summary. Defaults to 130. The summarization model limits the summary to a maximum of this length.
        do_sample : boolean, default=False
            Whether to use sampling during summary generation. Defaults to False. When True, the model uses a sampling technique for token selection.

    Returns:
        pd.DataFrame
            The dataFrame containing consolidated results from all models.
    """

    for model in models:
        print('Model :: {}'.format(model))
        logger.info('Model :: {}'.format(model))

        try:
            nlp = pipeline(task=task_name, model=model)
            input_args = {}

            if task_models.is_question_supported(task_name):
                input_args["question"] = question

            if task_name == "question-answering":
                input_args["context"] = input_data
                result = nlp(input_args)
            elif task_name == "document-question-answering":
                input_args["image"] = input_data
                result = nlp(input_args)
            elif task_name == "summarization":
                input_args["min_length"] = min_length
                input_args["max_length"] = max_length
                input_args["do_sample"] = do_sample
                result = nlp(input_data, **input_args)
            elif task_name == "sentiment-analysis":
                result = nlp(input_data)

            results_df = consolidate_results(result, results_df, model)
        except ValueError as e:
            print('Skipping model due to the error.')
            logger.error('An error occurred: %s', e)

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
            Results DataFrame with the consolidated result.
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
