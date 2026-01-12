import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_model(model, X_test, y_test, scaler):
    # Predict
    predictions = model.predict(X_test)

    # Inverse scaling
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Metrics
    mse = mean_squared_error(y_test_inv, predictions_inv)
    rmse = np.sqrt(mse)

    return mse, rmse, y_test_inv, predictions_inv
