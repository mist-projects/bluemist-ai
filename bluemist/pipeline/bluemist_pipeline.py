import os

from joblib import dump

pipeline_steps = {}
pipelines = {}

ARTIFACT_PATH = os.environ["ARTIFACT_PATH"]


def save_preprocessor(preprocessor):
    dump(preprocessor, ARTIFACT_PATH + '/' + 'preprocessor/preprocessor.joblib')
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
    dump(pipeline, ARTIFACT_PATH + '/' + 'models/' + estimator_name + '.joblib')
    print('pipelines', pipelines[estimator_name])
