import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Optional boto3 for S3 operations
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("Warning: boto3 not installed. S3 operations will be skipped.")

# --- Configuration ---
S3_BUCKET_NAME = "my-intelligent-scaling-data-bucket"
DATA_FILE_KEY = "multi_metric_data.csv"
MODEL_FILE_KEY = "lstm_model.pth"
LOCAL_DATA_PATH = "multi_metric_data.csv"
LOCAL_MODEL_PATH = "lstm_model.pth"

# Hyperparameters
SEQUENCE_LENGTH = 12
HIDDEN_SIZE = 50
NUM_LAYERS = 2
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 32

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

# --- 4. Define PyTorch LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
input_size = len(features)
output_size = 1
model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, output_size)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
print("Starting model training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

print("Model training complete.")

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")

# --- 5. Save the Trained Model ---
torch.save(model.state_dict(), LOCAL_MODEL_PATH)
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
