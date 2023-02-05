import os

from joblib import dump

pipeline_steps = {}
pipelines = {}

BLUEMIST_PATH = os.environ["BLUEMIST_PATH"]


def save_preprocessor(preprocessor):
    dump(preprocessor, BLUEMIST_PATH + '/' + 'artifacts/preprocessor/preprocessor.joblib')
    print('preprocessor', preprocessor)


def add_pipeline_step(estimator_name, pipeline_step):
    if estimator_name not in pipeline_steps:
        pipeline_steps[estimator_name] = []

    if estimator_name in pipeline_steps:
        steps = pipeline_steps[estimator_name]
        steps.append(pipeline_step)
        print('pipeline steps', steps)
        return steps


def save_pipeline(estimator_name, pipeline):
    pipelines[estimator_name] = pipeline
    dump(pipeline, BLUEMIST_PATH + '/' + 'artifacts/models/' + estimator_name + '.joblib')
    print('pipelines', pipelines[estimator_name])
