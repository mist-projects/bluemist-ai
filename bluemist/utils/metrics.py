import logging
import os
from logging import config
import numpy as np
from sklearn.metrics import *
import pandas as pd

BLUEMIST_PATH = os.environ["BLUEMIST_PATH"]
config.fileConfig(BLUEMIST_PATH + '/' + 'logging.config')
logger = logging.getLogger("bluemist")

default_regression_metrics = [mean_absolute_error, mean_squared_error, r2_score]
all_regression_metrics = [*default_regression_metrics, explained_variance_score, max_error, mean_squared_log_error,
                          median_absolute_error, mean_absolute_percentage_error, mean_poisson_deviance,
                          mean_gamma_deviance, mean_tweedie_deviance, d2_tweedie_score, mean_pinball_loss]

classification_metrics = []
all_classification_metrics = []


class scoringStrategy:
    y_true = None
    y_pred = None
    metrics_requested = None
    metrics_to_be_returned = None

    def __init__(self, y_true, y_pred, metrics):
        self.y_true = y_true
        self.y_pred = y_pred
        self.metrics_requested = metrics
        self.metrics_to_be_returned = []

    def getStats(self):

        rows = []
        row = []

        logger.info('Metrics requested :: {}'.format(self.metrics_requested))

        indexes_with_nan_predictions = np.argwhere(np.isnan(self.y_pred))
        self.y_true = np.delete(self.y_true, indexes_with_nan_predictions)
        self.y_pred = np.delete(self.y_pred, indexes_with_nan_predictions)
        logger.info('Indexes with NaN predictions to be removed :: {}'.format(indexes_with_nan_predictions))

        if isinstance(self.metrics_requested, str) and self.metrics_requested == 'default':
            for metric in default_regression_metrics:
                self.metrics_to_be_returned.append(metric.__name__)
                try:
                    row.append(metric(self.y_true, self.y_pred))
                except Exception as e:
                    row.append(str(e))
                    logger.error('Exception occurred while computing metric :: {}'.format(str(e)), exc_info=True)
        elif isinstance(self.metrics_requested, str) and self.metrics_requested == 'all':
            for metric in all_regression_metrics:
                self.metrics_to_be_returned.append(metric.__name__)
                try:
                    row.append(metric(self.y_true, self.y_pred))
                except Exception as e:
                    row.append(str(e))
                    logger.error('Exception occurred while computing metric :: {}'.format(str(e)), exc_info=True)
        elif isinstance(self.metrics_requested, list) and len(self.metrics_requested) > 0:
            for metric in list:
                self.metrics_to_be_returned.append(metric.__name__)
                try:
                    row.append(metric(self.y_true, self.y_pred))
                except Exception as e:
                    row.append(str(e))
                    logger.error('Exception occurred while computing metric :: {}'.format(str(e)), exc_info=True)

        rows.append(row)
        stats_df = pd.DataFrame(rows, columns=self.metrics_to_be_returned)
        return stats_df
