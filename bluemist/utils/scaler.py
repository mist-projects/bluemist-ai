
# Author: Shashank Agrawal
# License: MIT
# Version: 0.1.2
# Email: dew@bluemist-ai.one
# Created: Jun 22, 2022
# Last modified: June 11, 2023

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

available_scalers = {
    'StandardScaler': StandardScaler,
    'MinMaxScaler': MinMaxScaler,
    'MaxAbsScaler': MaxAbsScaler,
    'RobustScaler': RobustScaler
}


def get_scaler(scaling_strategy):
    return available_scalers.get(scaling_strategy, None)()
