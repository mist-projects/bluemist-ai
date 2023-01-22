from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

available_scalers = ['StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler']


def getScaler(scaling_strategy):
    scaler = None

    if scaling_strategy == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
    elif scaling_strategy == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaling_strategy == 'RobustScaler':
        scaler = RobustScaler()
    elif scaling_strategy == 'StandardScaler':
        scaler = StandardScaler()

    return scaler
