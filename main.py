
from src.data_loader import load_dataset
from src.config import DATA_PATH
from src.preprocessing import preprocess_dates
from src.config import TARGET_COLUMN
from src.preprocessing import train_test_split_and_scale
from src.sequence_generator import create_sequences
from src.models import build_simplernn
from src.evaluate import evaluate_model
from src.models import build_lstm
from sklearn.metrics import mean_squared_error
import numpy as np
from src.visualization import plot_predictions
from src.preprocessing import preprocess_dates, train_test_split_and_scale
import pandas as pd
import os


def multi_step_prediction(model, last_sequence, scaler, n_steps):
    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(n_steps):
        pred = model.predict(current_sequence, verbose=0)
        predictions.append(pred[0, 0])

        # Update sequence
        current_sequence = np.append(
            current_sequence[:, 1:, :],
            [[[pred[0, 0]]]],
            axis=1
        )

    # Inverse scale predictions
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return predictions

# Step 1: Load dataset
df = load_dataset(DATA_PATH)

# Basic validation
print("Dataset Loaded Successfully\n")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())



# Step 2: Date handling & sorting
df = preprocess_dates(df)

# Select target column
target_series = df[[TARGET_COLUMN]]

print("\nAfter Date Processing:")
print(target_series.head())
print("\nIndex Type:", type(target_series.index))

# Step 3: Train-test split & scaling
train_scaled, test_scaled, scaler = train_test_split_and_scale(target_series)

print("\nTrain Data Shape:", train_scaled.shape)
print("Test Data Shape:", test_scaled.shape)

LOOKBACK = 60

# Step 4: Create sequences
X_train, y_train = create_sequences(train_scaled, LOOKBACK)
X_test, y_test = create_sequences(test_scaled, LOOKBACK)

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Step 5: Build SimpleRNN
simplernn_model = build_simplernn(
    input_shape=(X_train.shape[1], X_train.shape[2])
)

simplernn_model.summary()

# Train SimpleRNN
history_rnn = simplernn_model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Step 6: Evaluate SimpleRNN
rnn_mse, rnn_rmse, y_actual, y_pred = evaluate_model(
    simplernn_model,
    X_test,
    y_test,
    scaler
)

print("\nSimpleRNN Evaluation:")
print("MSE:", rnn_mse)
print("RMSE:", rnn_rmse)

# Step 7: Build LSTM
lstm_model = build_lstm(
    input_shape=(X_train.shape[1], X_train.shape[2])
)

lstm_model.summary()

# Train LSTM
history_lstm = lstm_model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# =========================
# LSTM Evaluation
# =========================

lstm_predictions = lstm_model.predict(X_test)

# Inverse scale
lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))


lstm_mse = mean_squared_error(y_test_actual, lstm_predictions)
lstm_rmse = np.sqrt(lstm_mse)

# Use last 60 days from test data
last_60_days = X_test[-1].reshape(1, 60, 1)


print("\nLSTM Evaluation:")
print("MSE:", lstm_mse)
print("RMSE:", lstm_rmse)

# Inverse transform predictions
lstm_predictions = scaler.inverse_transform(lstm_predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

plot_predictions(
    actual_prices=y_test_actual,
    predicted_prices=lstm_predictions,
    title="LSTM Model: Actual vs Predicted Tesla Stock Price"
)


print("\nModel Comparison:")
print(f"SimpleRNN RMSE: {rnn_rmse:.2f}")
print(f"LSTM RMSE: {lstm_rmse:.2f}")

if lstm_rmse < rnn_rmse:
    print("✅ LSTM performs better than SimpleRNN")
else:
    print("✅ SimpleRNN performs better than LSTM")

# Step 8: Multi-step prediction with LSTM
pred_1_day = multi_step_prediction(
    lstm_model, last_60_days, scaler, n_steps=1
)

pred_5_days = multi_step_prediction(
    lstm_model, last_60_days, scaler, n_steps=5
)

pred_10_days = multi_step_prediction(
    lstm_model, last_60_days, scaler, n_steps=10
)

print("\n📈 Future Stock Price Predictions (Using LSTM):")

print(f"🔹 1 Day Ahead Prediction: ${pred_1_day[0][0]:.2f}")

print("\n🔹 5 Days Ahead Predictions:")
for i, price in enumerate(pred_5_days, 1):
    print(f"Day {i}: ${price[0]:.2f}")

print("\n🔹 10 Days Ahead Predictions:")
for i, price in enumerate(pred_10_days, 1):
    print(f"Day {i}: ${price[0]:.2f}")


'''
from src.data_loader import load_data
from src.preprocessing import preprocess_dates, scale_data, train_test_split
from src.sequence_generator import create_sequences
from src.models import build_simplernn, build_lstm
from src.evaluate import evaluate_model
import numpy as np

# =========================
# CONFIG
# =========================
DATA_PATH = "data/raw/TSLA.csv"
LOOKBACK = 60
EPOCHS = 20
BATCH_SIZE = 32

os.makedirs("models", exist_ok=True)


# =============================
# STEP 1: Load Dataset
# =============================
data = pd.read_csv(DATA_PATH)
data = preprocess_dates(data)

# Use ONLY Close price
data = data[['Close']]

print("✅ Data loaded successfully")


# =============================
# STEP 2: Train-Test Split & Scaling
# =============================
train_scaled, test_scaled, scaler = train_test_split_and_scale(data)

print("✅ Data scaled successfully")


# =============================
# STEP 3: Create Sequences
# =============================
X_train, y_train = create_sequences(train_scaled, LOOKBACK)
X_test, y_test = create_sequences(test_scaled, LOOKBACK)

print("✅ Sequences created")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# =============================
# STEP 4: Build Models
# =============================
input_shape = (X_train.shape[1], X_train.shape[2])

simplernn_model = build_simplernn(input_shape)
lstm_model = build_lstm(input_shape)

print("✅ Models built")


# =============================
# STEP 5: Train Models
# =============================
simplernn_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

lstm_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

print("✅ Models trained")


# =============================
# STEP 6: Save Models
# =============================
simplernn_model.save("models/simplernn_model.h5")
lstm_model.save("models/lstm_model.h5")

print("✅ Models saved")


# =============================
# STEP 7: Evaluate Models
# =============================
rnn_mse, rnn_rmse, y_test_inv, rnn_preds = evaluate_model(
    simplernn_model, X_test, y_test, scaler
)

lstm_mse, lstm_rmse, _, lstm_preds = evaluate_model(
    lstm_model, X_test, y_test, scaler
)

print("\n📊 Model Performance")
print(f"SimpleRNN RMSE: {rnn_rmse:.2f}")
print(f"LSTM RMSE: {lstm_rmse:.2f}")

if lstm_rmse < rnn_rmse:
    print("✅ LSTM performs better than SimpleRNN")
else:
    print("✅ SimpleRNN performs better than LSTM")


# =============================
# STEP 8: Predict Next Day Price
# =============================
last_sequence = test_scaled[-LOOKBACK:]
last_sequence = last_sequence.reshape(1, LOOKBACK, 1)

next_day_price = lstm_model.predict(last_sequence)
next_day_price = scaler.inverse_transform(next_day_price)

print(
    f"\n📈 Predicted Next Day Closing Price (USD): "
    f"${next_day_price[0][0]:.2f}"
)

'''