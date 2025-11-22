import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
LOCAL_MODEL_PATH = "intelligent_cloud_scaling_system/Trained Models/lstm_model.pth"
LOCAL_DATA_PATH = "intelligent_cloud_scaling_system/Data Files/multi_metric_data.csv"
SEQUENCE_LENGTH = 12
HIDDEN_SIZE = 50
NUM_LAYERS = 2

# Define the LSTM model architecture
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

def create_sequences(data, sequence_length):
    """Creates sequences for LSTM input"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

def evaluate_model():
    """Evaluate the trained LSTM model and print accuracy metrics"""
    
    print("="*60)
    print("INTELLIGENT CLOUD SCALING - MODEL EVALUATION REPORT")
    print("="*60)
    
    # Load data
    print("\n📊 Loading data...")
    df = pd.read_csv(LOCAL_DATA_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp')
    
    # Check column names and use appropriate CPU column
    cpu_column = 'cpu' if 'cpu' in df.columns else 'cpu_utilization'
    features = [cpu_column, 'network_in', 'request_count', 'is_sale_active']
    
    print(f"   - Total data points: {len(df)}")
    print(f"   - Features used: {features}")
    
    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    # Create sequences
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
    print(f"   - Sequences created: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Testing samples: {len(X_test)}")
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Load model
    print("\n🤖 Loading trained model...")
    input_size = len(features)
    output_size = 1
    model = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, output_size)
    model.load_state_dict(torch.load(LOCAL_MODEL_PATH))
    model.eval()
    print("   ✓ Model loaded successfully")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    with torch.no_grad():
        train_predictions = model(X_train_tensor).numpy()
        test_predictions = model(X_test_tensor).numpy()
    
    # Inverse transform to get actual values
    # Create dummy arrays for inverse transform
    train_dummy = np.zeros((len(train_predictions), len(features)))
    train_dummy[:, 0] = train_predictions.flatten()
    train_pred_actual = scaler.inverse_transform(train_dummy)[:, 0]
    
    test_dummy = np.zeros((len(test_predictions), len(features)))
    test_dummy[:, 0] = test_predictions.flatten()
    test_pred_actual = scaler.inverse_transform(test_dummy)[:, 0]
    
    train_actual_dummy = np.zeros((len(y_train), len(features)))
    train_actual_dummy[:, 0] = y_train
    train_actual = scaler.inverse_transform(train_actual_dummy)[:, 0]
    
    test_actual_dummy = np.zeros((len(y_test), len(features)))
    test_actual_dummy[:, 0] = y_test
    test_actual = scaler.inverse_transform(test_actual_dummy)[:, 0]
    
    # Calculate metrics
    print("\n" + "="*60)
    print("📈 ACCURACY METRICS")
    print("="*60)
    
    # Training metrics
    train_mse = mean_squared_error(train_actual, train_pred_actual)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(train_actual, train_pred_actual)
    train_r2 = r2_score(train_actual, train_pred_actual)
    train_mape = np.mean(np.abs((train_actual - train_pred_actual) / train_actual)) * 100
    
    print("\n🎯 TRAINING SET METRICS:")
    print(f"   • Mean Squared Error (MSE):     {train_mse:.4f}")
    print(f"   • Root Mean Squared Error (RMSE): {train_rmse:.4f}")
    print(f"   • Mean Absolute Error (MAE):    {train_mae:.4f}")
    print(f"   • R² Score:                     {train_r2:.4f}")
    print(f"   • Mean Absolute % Error (MAPE): {train_mape:.2f}%")
    print(f"   • Accuracy (100 - MAPE):        {100 - train_mape:.2f}%")
    
    # Testing metrics
    test_mse = mean_squared_error(test_actual, test_pred_actual)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_actual, test_pred_actual)
    test_r2 = r2_score(test_actual, test_pred_actual)
    test_mape = np.mean(np.abs((test_actual - test_pred_actual) / test_actual)) * 100
    
    print("\n🎯 TESTING SET METRICS:")
    print(f"   • Mean Squared Error (MSE):     {test_mse:.4f}")
    print(f"   • Root Mean Squared Error (RMSE): {test_rmse:.4f}")
    print(f"   • Mean Absolute Error (MAE):    {test_mae:.4f}")
    print(f"   • R² Score:                     {test_r2:.4f}")
    print(f"   • Mean Absolute % Error (MAPE): {test_mape:.2f}%")
    print(f"   • Accuracy (100 - MAPE):        {100 - test_mape:.2f}%")
    
    # Additional analysis
    print("\n" + "="*60)
    print("📊 PREDICTION ANALYSIS")
    print("="*60)
    
    print(f"\n📍 Test Set Statistics:")
    print(f"   • Actual CPU Range:      {test_actual.min():.2f}% - {test_actual.max():.2f}%")
    print(f"   • Predicted CPU Range:   {test_pred_actual.min():.2f}% - {test_pred_actual.max():.2f}%")
    print(f"   • Actual Mean CPU:       {test_actual.mean():.2f}%")
    print(f"   • Predicted Mean CPU:    {test_pred_actual.mean():.2f}%")
    
    # Prediction errors
    errors = test_actual - test_pred_actual
    print(f"\n⚠️  Prediction Errors:")
    print(f"   • Mean Error:            {errors.mean():.4f}")
    print(f"   • Std Deviation:         {errors.std():.4f}")
    print(f"   • Max Overestimation:    {errors.min():.4f}")
    print(f"   • Max Underestimation:   {errors.max():.4f}")
    
    # Sample predictions
    print("\n" + "="*60)
    print("🔍 SAMPLE PREDICTIONS (Last 10 Test Samples)")
    print("="*60)
    print(f"\n{'Index':<8} {'Actual CPU':<15} {'Predicted CPU':<15} {'Error':<10}")
    print("-" * 60)
    
    for i in range(max(0, len(test_actual) - 10), len(test_actual)):
        actual = test_actual[i]
        predicted = test_pred_actual[i]
        error = actual - predicted
        print(f"{i:<8} {actual:<15.2f} {predicted:<15.2f} {error:<10.2f}")
    
    # Model performance summary
    print("\n" + "="*60)
    print("✅ MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    if test_r2 > 0.9:
        performance = "Excellent"
        emoji = "🌟"
    elif test_r2 > 0.8:
        performance = "Very Good"
        emoji = "⭐"
    elif test_r2 > 0.7:
        performance = "Good"
        emoji = "👍"
    else:
        performance = "Needs Improvement"
        emoji = "⚠️"
    
    print(f"\n{emoji} Overall Performance: {performance}")
    print(f"   • R² Score indicates the model explains {test_r2*100:.1f}% of variance")
    print(f"   • Average prediction error: ±{test_mae:.2f}% CPU utilization")
    print(f"   • Model accuracy: {100 - test_mape:.2f}%")
    
    print("\n" + "="*60)
    print("✨ Evaluation Complete!")
    print("="*60)
    
    return {
        'train_metrics': {
            'mse': train_mse,
            'rmse': train_rmse,
            'mae': train_mae,
            'r2': train_r2,
            'mape': train_mape,
            'accuracy': 100 - train_mape
        },
        'test_metrics': {
            'mse': test_mse,
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2,
            'mape': test_mape,
            'accuracy': 100 - test_mape
        }
    }

if __name__ == "__main__":
    metrics = evaluate_model()
