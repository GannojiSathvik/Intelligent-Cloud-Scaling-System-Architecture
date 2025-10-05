import pandas as pd
import numpy as np
import boto3
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

# --- Configuration ---
S3_BUCKET_NAME = "my-intelligent-scaling-data-bucket"
DATA_FILE_KEY = "multi_metric_data.csv"
MODEL_FILE_KEY = "lstm_model.h5"
LOCAL_DATA_PATH = "multi_metric_data.csv"
LOCAL_MODEL_PATH = "lstm_model.h5"

# --- 1. Download Data from S3 (or use local) ---
# In a real scenario, you'd uncomment this. For local testing, we use the generated file.
# s3_client = boto3.client('s3')
# try:
#     s3_client.download_file(S3_BUCKET_NAME, DATA_FILE_KEY, LOCAL_DATA_PATH)
#     print(f"Downloaded '{DATA_FILE_KEY}' from S3 bucket '{S3_BUCKET_NAME}'.")
# except Exception as e:
#     print(f"Failed to download from S3. Ensure the file exists and bucket name is correct. Error: {e}")
#     exit()

df = pd.read_csv(LOCAL_DATA_PATH, parse_dates=['timestamp'], index_col='timestamp')
print("Data loaded successfully.")

# --- 2. Preprocess Data ---
# We want to predict 'cpu_utilization', so it's our target.
features = ['cpu_utilization', 'network_in', 'request_count', 'is_sale_active']
data = df[features].values

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# We need a separate scaler for the target variable ('cpu_utilization') to inverse transform the predictions later
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit_transform(data[:, 0:1]) # Only fit on the CPU column

# --- 3. Create Sequences for LSTM ---
SEQUENCE_LENGTH = 12  # Use last 12 data points (60 minutes) to predict the next one
X, y = [], []

for i in range(SEQUENCE_LENGTH, len(scaled_data)):
    X.append(scaled_data[i-SEQUENCE_LENGTH:i])
    y.append(scaled_data[i, 0]) # The target is the next 'cpu_utilization'

X, y = np.array(X), np.array(y)

# Reshape X for LSTM [samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], len(features)))

print(f"Created {X.shape[0]} sequences of length {SEQUENCE_LENGTH}.")

# --- 4. Define and Train the LSTM Model ---
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Use EarlyStopping to prevent overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X, y,
    epochs=5, # Reduced for a quick demonstration
    batch_size=32,
    validation_split=0.2,
    # callbacks=[early_stopping],
    verbose=1
)

print("Model training complete.")

# --- 5. Save the Trained Model ---
model.save(LOCAL_MODEL_PATH)
print(f"Model saved locally to '{LOCAL_MODEL_PATH}'.")

# --- 6. Upload Model to S3 ---
# In a real scenario, you'd uncomment this.
# try:
#     s3_client.upload_file(LOCAL_MODEL_PATH, S3_BUCKET_NAME, MODEL_FILE_KEY)
#     print(f"Uploaded '{MODEL_FILE_KEY}' to S3 bucket '{S3_BUCKET_NAME}'.")
# except Exception as e:
#     print(f"Failed to upload model to S3. Error: {e}")

# Clean up local model file
# if os.path.exists(LOCAL_MODEL_PATH):
#     os.remove(LOCAL_MODEL_PATH)
