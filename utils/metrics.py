import math

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



class scoingStrategy:
    y_true = None
    y_pred = None

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def get_r2_score(self):
        return r2_score(self.y_true, self.y_pred)

    def get_mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def get_mse(self):
        return mean_squared_error(self.y_true, self.y_pred)

    def get_rmse(self):
        mse = self.get_mse()
        return math.sqrt(mse)

