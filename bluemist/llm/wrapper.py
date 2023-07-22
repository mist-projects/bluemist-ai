# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.3
# Email: dew@bluemist-ai.one
# Created: Jul 17, 2023
# Last modified: June 22, 2023

from transformers import pipeline
import pandas as pd
from task_models import TaskModels

# Instantiate the TaskModels class
task_models = TaskModels()


def deploy_model(task, model, context, question=None):
    evaluate_models(task, context, question, override_models=model, limit=1)


def perform_task(task, model, context, question=None):
    evaluate_models(task, context, question, override_models=model, limit=1)


def evaluate_models(task, context, question=None, override_models=None, limit=None):
    """
        **Performs the task on the given dataset, evaluate the models and returns comparison metrics**

        task : {'information-extraction', 'named-entity-recognition', 'question-answering', 'sentiment-analysis', 'summarization', 'text-classification'}
            task to be performed.
        context : str
            Text or information used by the model to perform specific NLP tasks.
        question : str, default=None
            Specific query or question provided as input to the model for question-answering tasks. The model uses this question to find the relevant answer within the provided context.
        override_models : str or list, default=None
            Provide additional models not part of the pre-configured list
        limit : int, default=None
            Limits the models to be compared
    """

    # Retrieve the pipeline task name for the given task
    task_name = task_models.get_task_name(task)
    print(task_models.get_all_tasks())

    if task_name is None:
        print(f"Unsupported task: {task}")
        return

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=["Model", "Answer", "Score"])

    models = []
    if isinstance(override_models, str):
        models.append(override_models)
    elif isinstance(override_models, list):
        models.extend(override_models)

    models.extend(task_models.get_models_for_task(task))
    num_of_models = len(models)

    if limit is not None and 0 < limit <= num_of_models:
        models = models[:limit]

    # Iterate through all available models for the task
    for model in models:
        print(f"Model: {model}")

        # Create the pipeline for the specific task
        nlp = pipeline(task=task_name, model=model)

        # Perform the task using the current model
        if task_models.is_question_supported(task):
            result = nlp(context=context, question=question)
        else:
            result = nlp(context=context)

        # Extract the answer and score from the result
        score = result['score']
        answer = result['answer']

        results_list = [{'Model': model, 'Score': score, 'Answer': answer}]
        new_rows_df = pd.DataFrame(results_list)

        # Store the results in the DataFrame
        results_df = pd.concat([results_df, new_rows_df], ignore_index=True)
        print(results_df)

    return results_df
