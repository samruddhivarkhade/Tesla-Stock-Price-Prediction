import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def preprocess_dates(df):
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by Date (CRITICAL for time series)
    df = df.sort_values('Date')

    # Set Date as index
    df.set_index('Date', inplace=True)

    return df

def train_test_split_and_scale(data, train_ratio=0.8):
    # Convert to numpy array
    values = data.values

    # Time-based split
    train_size = int(len(values) * train_ratio)
    train_data = values[:train_size]
    test_data = values[train_size:]

    # Scaling (fit ONLY on training data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    return train_scaled, test_scaled, scaler