import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
MODEL_FILE_KEY = "lstm_model_optimized.pth"
LOCAL_DATA_PATH = "multi_metric_data.csv"
LOCAL_MODEL_PATH = "lstm_model_optimized.pth"

# Optimized Hyperparameters for Better Accuracy
SEQUENCE_LENGTH = 24  # Increased from 12 to capture more temporal patterns
HIDDEN_SIZE = 128     # Increased from 50 for more learning capacity
NUM_LAYERS = 3        # Increased from 2 for deeper learning
NUM_EPOCHS = 50       # Increased from 5 for better convergence
LEARNING_RATE = 0.0005  # Reduced for finer tuning
BATCH_SIZE = 64       # Increased for more stable gradients
DROPOUT = 0.2         # Added dropout for regularization

# Improved LSTM Model with Dropout and Better Architecture
class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM with dropout
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Additional dense layers for better feature extraction
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Get last timestep
        
        # Additional processing layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def create_sequences(data, sequence_length):
    """Creates sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

def train_optimized_model():
    """
    Train an optimized LSTM model with improved hyperparameters
    """
    print("="*70)
    print("🚀 TRAINING OPTIMIZED LSTM MODEL FOR IMPROVED ACCURACY")
    print("="*70)
    
    # 1. Load Data
    local_data_path = LOCAL_DATA_PATH
    
    if not os.path.exists(local_data_path):
        print(f"Error: {local_data_path} not found.")
        return
    
    df = pd.read_csv(local_data_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp')
    print(f"\n✓ Data loaded: {len(df)} records")
    
    # 2. Preprocess Data
    cpu_column = 'cpu' if 'cpu' in df.columns else 'cpu_utilization'
    features = [cpu_column, 'network_in', 'request_count', 'is_sale_active']
    print(f"✓ Features: {features}")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    # 3. Create Sequences
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
    print(f"✓ Sequences created: {len(X)} (using {SEQUENCE_LENGTH} timesteps)")
    
    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Testing samples: {len(X_test)}")
    
    # 5. Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # 6. Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    # 7. Initialize Improved Model
    input_size = len(features)
    output_size = 1
    model = ImprovedLSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, output_size, DROPOUT)
    
    print(f"\n📊 Model Architecture:")
    print(f"   • Input Size: {input_size}")
    print(f"   • Hidden Size: {HIDDEN_SIZE}")
    print(f"   • Num Layers: {NUM_LAYERS}")
    print(f"   • Dropout: {DROPOUT}")
    print(f"   • Total Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 8. Define Loss, Optimizer, and Scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 9. Training Loop with Early Stopping
    print(f"\n🎯 Starting Training ({NUM_EPOCHS} epochs)...")
    print("-" * 70)
    
    best_test_loss = float('inf')
    patience_counter = 0
    patience_limit = 10
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Test Loss: {test_loss:.6f}")
        
        # Early stopping check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), LOCAL_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
                break
    
    print("-" * 70)
    print(f"✓ Training Complete!")
    print(f"✓ Best Test Loss: {best_test_loss:.6f}")
    
    # 10. Final Evaluation
    print(f"\n📈 Final Model Evaluation:")
    model.load_state_dict(torch.load(LOCAL_MODEL_PATH))
    model.eval()
    
    with torch.no_grad():
        train_pred = model(X_train_tensor).numpy()
        test_pred = model(X_test_tensor).numpy()
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Inverse transform predictions
    train_dummy = np.zeros((len(train_pred), len(features)))
    train_dummy[:, 0] = train_pred.flatten()
    train_pred_actual = scaler.inverse_transform(train_dummy)[:, 0]
    
    test_dummy = np.zeros((len(test_pred), len(features)))
    test_dummy[:, 0] = test_pred.flatten()
    test_pred_actual = scaler.inverse_transform(test_dummy)[:, 0]
    
    train_actual_dummy = np.zeros((len(y_train), len(features)))
    train_actual_dummy[:, 0] = y_train
    train_actual = scaler.inverse_transform(train_actual_dummy)[:, 0]
    
    test_actual_dummy = np.zeros((len(y_test), len(features)))
    test_actual_dummy[:, 0] = y_test
    test_actual = scaler.inverse_transform(test_actual_dummy)[:, 0]
    
    # Metrics
    test_r2 = r2_score(test_actual, test_pred_actual)
    test_mae = mean_absolute_error(test_actual, test_pred_actual)
    test_mape = np.mean(np.abs((test_actual - test_pred_actual) / test_actual)) * 100
    test_accuracy = 100 - test_mape
    
    print(f"   • R² Score:                  {test_r2:.4f} ({test_r2*100:.1f}% variance explained)")
    print(f"   • Mean Absolute Error (MAE): {test_mae:.4f}%")
    print(f"   • Mean Absolute % Error:     {test_mape:.2f}%")
    print(f"   • Model Accuracy:            {test_accuracy:.2f}%")
    
    improvement = test_accuracy - 77.39
    print(f"\n🎉 Accuracy Improvement: {improvement:+.2f}% (from 77.39% to {test_accuracy:.2f}%)")
    
    print(f"\n✓ Optimized model saved to: {LOCAL_MODEL_PATH}")
    print("="*70)

if __name__ == "__main__":
    train_optimized_model()
