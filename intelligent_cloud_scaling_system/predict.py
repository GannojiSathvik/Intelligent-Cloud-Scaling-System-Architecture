import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# --- Configuration ---
LOCAL_MODEL_PATH = "lstm_model.h5"
LOCAL_DATA_PATH = "multi_metric_data.csv"

def run_prediction():
    """
    Loads the trained model and latest data to make a single prediction.
    """
    # --- 1. Check if Model and Data exist ---
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"Error: Model file '{LOCAL_MODEL_PATH}' not found.")
        print("Please wait for the training script (train_model.py) to complete.")
        return

    if not os.path.exists(LOCAL_DATA_PATH):
        print(f"Error: Data file '{LOCAL_DATA_PATH}' not found.")
        return

    print("Loading model and data...")
    model = load_model(LOCAL_MODEL_PATH)
    df = pd.read_csv(LOCAL_DATA_PATH)

    # --- 2. Get the last sequence of data ---
    if len(df) < 12:
        print("Error: Not enough data to create a sequence (need at least 12 data points).")
        return
    
    latest_data = df.tail(12)
    print("Using the last 12 data points to make a prediction:")
    print(latest_data)

    # --- 3. Preprocess the data ---
    features = ['cpu_utilization', 'network_in', 'request_count', 'is_sale_active']
    sequence_data = latest_data[features].values

    # Use the same scaling logic as in the training and scaling scripts
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df[features].values) # Fit on the whole dataset to get the correct range
    scaled_sequence = scaler.transform(sequence_data)

    # Reshape for the LSTM model
    input_sequence = np.reshape(scaled_sequence, (1, 12, len(features)))

    # --- 4. Make and print the prediction ---
    predicted_scaled_cpu = model.predict(input_sequence)[0][0]

    # Inverse transform to get the actual CPU percentage
    dummy_array = np.zeros((1, len(features)))
    dummy_array[0, 0] = predicted_scaled_cpu
    predicted_cpu = scaler.inverse_transform(dummy_array)[0, 0]

    print("\n" + "="*30)
    print(f"  PREDICTED CPU UTILIZATION (next 5 mins): {predicted_cpu:.2f}%")
    print("="*30)

if __name__ == "__main__":
    run_prediction()
