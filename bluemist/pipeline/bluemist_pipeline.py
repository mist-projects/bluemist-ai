
__author__ = "Shashank Agrawal"
__license__ = "MIT"
__version__ = "0.1.1"
__email__ = "dew@bluemist-ai.one"


import logging
import os
from logging import config

import joblib
from joblib import dump

pipeline_steps = {}
pipelines = {}

BLUEMIST_PATH = os.environ["BLUEMIST_PATH"]
config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")


def save_preprocessor(preprocessor):
    preprocessor_disk_location = BLUEMIST_PATH + '/' + 'artifacts/preprocessor/preprocessor.joblib'
    logger.info('Saving preprocessor to disk on :: {}'.format(preprocessor_disk_location))
    dump(preprocessor, preprocessor_disk_location)
    logger.debug('Preprocessor column transformer object :: {}'.format(preprocessor))


def add_pipeline_step(estimator_name, pipeline_step):
    if estimator_name not in pipeline_steps:
        pipeline_steps[estimator_name] = []

    if estimator_name in pipeline_steps:
        steps = pipeline_steps[estimator_name]
        steps.append(pipeline_step)
        logger.debug('Model pipeline steps :: {}'.format(steps))
        return steps


def save_model_pipeline(estimator_name, pipeline):
    model_pipeline_disk_location = BLUEMIST_PATH + '/' + 'artifacts/models/' + estimator_name + '.joblib'
    pipelines[estimator_name] = pipeline
    logger.info('Saving model pipeline to disk :: {}'.format(model_pipeline_disk_location))
    dump(pipeline, model_pipeline_disk_location)
    logger.info('Model pipeline object :: {}'.format(pipeline))


def clear_all_model_pipelines():
    global pipeline_steps, pipelines
    pipeline_steps = {}
    pipelines = {}


def get_model_pipeline(estimator_name):
    logger.info('Getting model pipeline for {}'.format(estimator_name))
    pipeline = pipelines[estimator_name]
    return pipeline
