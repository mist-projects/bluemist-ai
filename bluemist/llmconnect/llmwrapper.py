from transformers import pipeline
import pandas as pd
from task_models import TaskModels

# Instantiate the TaskModels class
task_models = TaskModels()


def evaluate_models(task, context, question=None):
    # Retrieve the pipeline task name for the given task
    task_name = task_models.get_task_name(task)

    if task_name is None:
        print(f"Unsupported task: {task}")
        return

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=["Model", "Answer", "Score"])

    # Iterate through all available models for the task
    for model in task_models.get_models_for_task(task):
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

        # Store the results in the DataFrame
        results_df = results_df.concat({'Model': model, 'Score': score, 'Answer': answer}, ignore_index=True)
        print(results_df)

    return results_df
