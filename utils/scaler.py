from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler


def getScaler(scaling_type):
    scaler = None

    if scaling_type == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
    elif scaling_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaling_type == 'RobustScaler':
        scaler = RobustScaler()
    elif scaling_type == 'StandardScaler':
        scaler = StandardScaler()

    return scaler
