import streamlit as st
import numpy as np
from src.data_loader import load_dataset
from src.preprocessing import preprocess_dates, train_test_split_and_scale
from src.sequence_generator import create_sequences
from src.models import build_simplernn, build_lstm
from src.evaluate import evaluate_model
from src.visualization import plot_predictions
from src.config import TARGET_COLUMN


st.set_page_config(page_title="Tesla Stock Predictor", layout="wide")
st.title("🚀 Tesla Stock Price Prediction")


# Sidebar
st.sidebar.header("Model Options")
model_choice = st.sidebar.selectbox("Select Model", ["SimpleRNN", "LSTM"])
epochs = st.sidebar.slider("Epochs", min_value=10, max_value=50, value=20, step=5)
lookback = st.sidebar.slider("Lookback Days", min_value=10, max_value=100, value=60, step=10)

# Load and display data
from src.data_loader import load_dataset
from src.config import DATA_PATH

df = load_dataset(DATA_PATH)
st.subheader("Dataset Overview")
st.dataframe(df.tail(10))

# Preprocessing
df = preprocess_dates(df)
target_series = df[[TARGET_COLUMN]]
train_scaled, test_scaled, scaler = train_test_split_and_scale(target_series)

# Sequence generation
X_train, y_train = create_sequences(train_scaled, lookback)
X_test, y_test = create_sequences(test_scaled, lookback)

# Model selection
input_shape = (X_train.shape[1], X_train.shape[2])
if model_choice == "SimpleRNN":
    model = build_simplernn(input_shape)
else:
    model = build_lstm(input_shape)

# Train model
st.write(f"Training {model_choice} for {epochs} epochs...")
model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)
st.success("✅ Training Complete!")

# Evaluate
mse, rmse, y_actual, y_pred = evaluate_model(model, X_test, y_test, scaler)
st.write(f"**MSE:** {mse:.2f}, **RMSE:** {rmse:.2f}")

# Plot actual vs predicted
st.subheader("📈 Actual vs Predicted Prices")
fig = plot_predictions(y_actual, y_pred)
st.pyplot(fig)

# Multi-day prediction
st.subheader("🔮 Future Predictions")
days_ahead = st.slider("Days ahead to predict", 1, 10, 5)
last_seq = X_test[-1].reshape(1, lookback, 1)
future_preds = []
current_seq = last_seq.copy()

for _ in range(days_ahead):
    pred = model.predict(current_seq)
    future_preds.append(pred[0, 0])
    current_seq = np.append(current_seq[:, 1:, :], [[[pred[0, 0]]]], axis=1)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
for i, price in enumerate(future_preds, 1):
    st.write(f"Day {i}: ${price[0]:.2f}")
