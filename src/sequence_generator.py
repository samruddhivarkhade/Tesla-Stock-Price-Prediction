import numpy as np

def create_sequences(data, lookback=60):
    X, y = [], []

    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshape X for RNN/LSTM: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y
