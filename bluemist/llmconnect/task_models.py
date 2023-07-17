# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.3
# Email: dew@bluemist-ai.one
# Created:  Jul 17, 2023
# Last modified: Jul 17, 2023

class TaskModels:
    """
    Class representing a collection of tasks and their associated models.

    This class provides methods to retrieve information about available tasks
    and models. It also allows accessing the pipeline task name for a given task.

    Attributes:
        tasks (dict): A dictionary mapping tasks to a list of associated models
            and their corresponding pipeline task names.

    Example usage:
    ```
    # Instantiate the TaskModels class
    task_models = TaskModels()

    # Retrieve a list of available tasks
    tasks = task_models.list_tasks()
    print("Available tasks:", tasks)

    # Retrieve the list of models for a specific task
    task = "question-answering"
    models = task_models.list_models(task)
    print(f"Models for {task} task:", models)

    # Retrieve the pipeline task name for a specific task
    task = "sentiment-analysis"
    task_name = task_models.get_task_name(task)
    print(f"Task name for {task} task:", task_name)
    ```
    """

    def __init__(self):
        """
        Initialize the TaskModels instance.

        This constructor initializes the `tasks` attribute, which contains the
        predefined tasks and their associated models.
        """
        self.tasks = {}
        self.populate_tasks()

    def populate_tasks(self):
        """
        Populates the tasks dictionary with available tasks and their associated models.
        """
        self.tasks = {
            "information-extraction": {
                "models": [
                    "bert-base-cased",
                    "roberta-large",
                    "electra-large",
                    "albert-base-v2",
                ],
                "task_name": "ner",
                "question_support": False,
            },
            "named-entity-recognition": {
                "models": [
                    "bert-base-cased",
                    "roberta-base",
                    "camembert-base",
                    "xlm-roberta-large",
                ],
                "task_name": "ner",
                "question_support": False,
            },
            "question-answering": {
                "models": [
                    "bert-large-uncased-whole-word-masking-finetuned-squad",
                    "distilbert-base-uncased-distilled-squad",
                    "roberta-large-squad2",
                    "xlnet-large-cased",
                ],
                "task_name": "question-answering",
                "question_support": True,
            },
            "sentiment-analysis": {
                "models": [
                    "bert-base-uncased",
                    "distilbert-base-uncased",
                    "roberta-base",
                    "xlm-roberta-base",
                ],
                "task_name": "sentiment-analysis",
                "question_support": False,
            },
            "summarization": {
                "models": [
                    "bart-large-cnn",
                    "pegasus-large",
                    "t5-base",
                    "t5-large",
                ],
                "task_name": "summarization",
                "question_support": False,
            },
            "text-classification": {
                "models": [
                    "bert-base-uncased",
                    "distilbert-base-uncased",
                    "roberta-base",
                    "xlm-roberta-base",
                ],
                "task_name": "text-classification",
                "question_support": False,
            },
        }

    def get_models_for_task(self, task):
        """
        Retrieves the available models for a given task.

        Args:
            task (str): The task for which to retrieve the models.

        Returns:
            list: A list of available models for the specified task.
        """
        if task in self.tasks:
            return self.tasks[task]['models']
        else:
            return []

    def get_all_tasks(self):
        """
        Retrieves all available tasks.

        Returns:
            list: A list of all available tasks.
        """
        return list(self.tasks.keys())

    def get_task_name(self, task):
        """
        Retrieve the pipeline task name for a given task.

        Args:
            task (str): The task for which to retrieve the pipeline task name.

        Returns:
            str or None: The pipeline task name associated with the given task,
                or None if the task is not found in the dictionary.
        """
        if task in self.tasks:
            return self.tasks[task]['task_name']
        else:
            return None

    def is_question_supported(self, task):
        """
        Check if the given task supports questions.

        Args:
            task (str): The task name to check.

        Returns:
            bool: True if the task supports questions, False otherwise.
        """
        if task in self.tasks:
            return self.tasks[task]["question_support"]
        else:
            return False


