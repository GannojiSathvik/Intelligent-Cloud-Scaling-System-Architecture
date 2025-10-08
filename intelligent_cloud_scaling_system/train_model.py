import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import boto3
import os

# Define S3 bucket and file paths
S3_BUCKET_NAME = 'intelligent-scaling-demo-sathvik'
DATA_FILE_KEY = 'multi_metric_data.csv'
MODEL_FILE_KEY = 'models/lstm_model.pth'

# Hyperparameters
SEQUENCE_LENGTH = 12 # Use 12 previous timesteps to predict the next
HIDDEN_SIZE = 50
NUM_LAYERS = 2
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 32

class LSTMModel(nn.Module):
    """
    PyTorch LSTM model for time series prediction.
    """
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
        out = self.fc(out[:, -1, :]) # Get the output from the last timestep
        return out

def create_sequences(data, sequence_length):
    """
    Creates sequences of data for LSTM training.

    Args:
        data (np.array): The input data.
        sequence_length (int): The length of each input sequence.

    Returns:
        tuple: A tuple containing input sequences (X) and target values (y).
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length, 0]) # Predict the first feature (CPU)
    return np.array(X), np.array(y)

def download_from_s3(bucket_name, key, local_path):
    """Downloads a file from S3."""
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, key, local_path)
        print(f"Successfully downloaded {key} from S3 bucket {bucket_name} to {local_path}")
    except Exception as e:
        print(f"Error downloading {key} from S3: {e}")
        raise

def upload_to_s3(bucket_name, local_path, key):
    """Uploads a file to S3."""
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, bucket_name, key)
        print(f"Successfully uploaded {local_path} to S3 bucket {bucket_name} as {key}")
    except Exception as e:
        print(f"Error uploading {local_path} to S3: {e}")
        raise

def train_model():
    """
    Main function to train the PyTorch LSTM model.
    """
    # 1. Load Data: Download the multi_metric_data.csv from S3.
    local_data_path = 'multi_metric_data.csv'
    # Ensure the directory exists for local_data_path if it's not in the current directory
    # For this example, we assume it's in the current directory or a path that exists.
    # If running locally, you might generate data first:
    # from generate_synthetic_data import generate_synthetic_data
    # synthetic_df = generate_synthetic_data()
    # synthetic_df.to_csv(local_data_path, index=False)

    # In a real AWS environment (e.g., SageMaker), you'd download from S3.
    # For local testing, you might skip S3 download if data is already local.
    # For this script, we'll assume S3 download is the primary method.
    try:
        download_from_s3(S3_BUCKET_NAME, DATA_FILE_KEY, local_data_path)
    except Exception:
        print(f"Could not download {DATA_FILE_KEY} from S3. Attempting to load from local path.")
        if not os.path.exists(local_data_path):
            print("Local data file not found. Please ensure data is available.")
            return

    df = pd.read_csv(local_data_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp')

    # 2. Preprocess Data
    features = ['cpu', 'network_in', 'request_count', 'is_sale_active']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])

    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Define the PyTorch LSTM Model
    input_size = len(features)
    output_size = 1 # Predicting CPU
    model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, output_size)

    # 4. Train the Model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

    # Evaluate on test set (optional, but good practice)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f"Test Loss: {test_loss.item():.4f}")

    # 5. Save the Model
    local_model_path = 'lstm_model.pth'
    torch.save(model.state_dict(), local_model_path)
    print(f"Model saved locally to {local_model_path}")

    # 6. Upload the Model to S3
    upload_to_s3(S3_BUCKET_NAME, local_model_path, MODEL_FILE_KEY)

    print("Training process finished.")

if __name__ == "__main__":
    train_model()
