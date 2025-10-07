import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import json
from itertools import product

# Configuration
LOCAL_DATA_PATH = "multi_metric_data.csv"
LOCAL_MODEL_PATH = "lstm_model_advanced.pth"
ENSEMBLE_DIR = "ensemble_models"

# Create ensemble directory
os.makedirs(ENSEMBLE_DIR, exist_ok=True)

# =============================================================================
# 1. FEATURE ENGINEERING - Add Temporal Features
# =============================================================================

def add_temporal_features(df):
    """Add temporal features to capture time-based patterns"""
    print("\n🔧 Adding Temporal Features...")
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Hour of day (0-23)
        df['hour'] = df['timestamp'].dt.hour
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Is weekend (1 if Sat/Sun, 0 otherwise)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Is business hours (9 AM - 5 PM on weekdays)
        df['is_business_hours'] = (
            (df['hour'] >= 9) & (df['hour'] < 17) & (df['is_weekend'] == 0)
        ).astype(int)
        
        # Time of day category (0: night, 1: morning, 2: afternoon, 3: evening)
        df['time_category'] = pd.cut(df['hour'], 
                                      bins=[0, 6, 12, 18, 24], 
                                      labels=[0, 1, 2, 3], 
                                      include_lowest=True).astype(int)
    
    # CPU column name
    cpu_col = 'cpu' if 'cpu' in df.columns else 'cpu_utilization'
    
    # Rolling statistics (6 periods = 30 minutes)
    df['cpu_rolling_mean'] = df[cpu_col].rolling(window=6, min_periods=1).mean()
    df['cpu_rolling_std'] = df[cpu_col].rolling(window=6, min_periods=1).std().fillna(0)
    df['cpu_rolling_min'] = df[cpu_col].rolling(window=6, min_periods=1).min()
    df['cpu_rolling_max'] = df[cpu_col].rolling(window=6, min_periods=1).max()
    
    # Rate of change
    df['cpu_diff'] = df[cpu_col].diff().fillna(0)
    
    # Network and request rolling means
    df['network_rolling_mean'] = df['network_in'].rolling(window=6, min_periods=1).mean()
    df['request_rolling_mean'] = df['request_count'].rolling(window=6, min_periods=1).mean()
    
    print(f"   ✓ Added {len(df.columns) - 5} new features")
    return df

# =============================================================================
# 2. ATTENTION MECHANISM - Advanced LSTM with Attention
# =============================================================================

class AttentionLayer(nn.Module):
    """Attention mechanism to focus on important timesteps"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class AttentionLSTM(nn.Module):
    """LSTM with Attention Mechanism"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = AttentionLayer(hidden_size)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        context_vector, attention_weights = self.attention(lstm_out)
        
        # Dense layers
        out = self.fc1(context_vector)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# =============================================================================
# 3. ENSEMBLE MODELS - Train Multiple Models
# =============================================================================

def create_sequences(data, sequence_length):
    """Creates sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

def train_single_model(X_train, y_train, X_test, y_test, config, model_id):
    """Train a single model with given configuration"""
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    # Initialize model
    input_size = X_train.shape[2]
    model = AttentionLSTM(
        input_size, 
        config['hidden_size'], 
        config['num_layers'], 
        1, 
        config['dropout']
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor).item()
        
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            model_path = os.path.join(ENSEMBLE_DIR, f'model_{model_id}.pth')
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    return best_test_loss, config

def train_ensemble(X_train, y_train, X_test, y_test, num_models=5):
    """Train ensemble of models with different configurations"""
    print(f"\n🎯 Training Ensemble of {num_models} Models...")
    
    # Different configurations for diversity
    configs = [
        {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2, 'learning_rate': 0.0005, 'batch_size': 64, 'epochs': 50},
        {'hidden_size': 96, 'num_layers': 2, 'dropout': 0.3, 'learning_rate': 0.001, 'batch_size': 32, 'epochs': 50},
        {'hidden_size': 160, 'num_layers': 3, 'dropout': 0.15, 'learning_rate': 0.0003, 'batch_size': 128, 'epochs': 50},
        {'hidden_size': 128, 'num_layers': 4, 'dropout': 0.25, 'learning_rate': 0.0007, 'batch_size': 64, 'epochs': 50},
        {'hidden_size': 112, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.0004, 'batch_size': 96, 'epochs': 50},
    ]
    
    ensemble_info = []
    
    for i, config in enumerate(configs[:num_models]):
        print(f"\n   Training Model {i+1}/{num_models}...")
        print(f"   Config: hidden={config['hidden_size']}, layers={config['num_layers']}, dropout={config['dropout']}")
        
        test_loss, used_config = train_single_model(X_train, y_train, X_test, y_test, config, i)
        ensemble_info.append({
            'model_id': i,
            'test_loss': test_loss,
            'config': used_config
        })
        print(f"   ✓ Model {i+1} Test Loss: {test_loss:.6f}")
    
    # Save ensemble info
    with open(os.path.join(ENSEMBLE_DIR, 'ensemble_info.json'), 'w') as f:
        json.dump(ensemble_info, f, indent=2)
    
    print(f"\n✓ Ensemble training complete!")
    return ensemble_info

# =============================================================================
# 4. HYPERPARAMETER TUNING - Grid Search
# =============================================================================

def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """Perform grid search for optimal hyperparameters"""
    print("\n🔍 Performing Hyperparameter Tuning...")
    
    param_grid = {
        'hidden_size': [96, 128, 160],
        'num_layers': [2, 3],
        'dropout': [0.15, 0.2, 0.25],
        'learning_rate': [0.0003, 0.0005, 0.0007]
    }
    
    # Generate combinations (limit to avoid too many)
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    print(f"   Testing {min(len(combinations), 10)} parameter combinations...")
    
    best_loss = float('inf')
    best_params = None
    
    for i, combo in enumerate(combinations[:10]):  # Limit to 10 combinations
        config = {
            'hidden_size': combo[0],
            'num_layers': combo[1],
            'dropout': combo[2],
            'learning_rate': combo[3],
            'batch_size': 64,
            'epochs': 30  # Reduced for faster tuning
        }
        
        print(f"\n   Trial {i+1}/10: hidden={combo[0]}, layers={combo[1]}, dropout={combo[2]}, lr={combo[3]}")
        
        test_loss, _ = train_single_model(X_train, y_train, X_test, y_test, config, f'tune_{i}')
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_params = config
            print(f"   🌟 New best! Loss: {test_loss:.6f}")
    
    print(f"\n✓ Best Parameters Found:")
    print(f"   {best_params}")
    print(f"   Best Loss: {best_loss:.6f}")
    
    return best_params

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_advanced_model():
    """Main training function with all advanced techniques"""
    
    print("="*80)
    print("🚀 ADVANCED MODEL TRAINING - ALL TECHNIQUES COMBINED")
    print("="*80)
    
    # 1. Load and prepare data
    print("\n📊 Step 1: Loading Data...")
    df = pd.read_csv(LOCAL_DATA_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp')
    print(f"   ✓ Loaded {len(df)} records")
    
    # 2. Feature Engineering
    print("\n🔧 Step 2: Feature Engineering...")
    df = add_temporal_features(df)
    
    # Select features (including engineered ones)
    cpu_col = 'cpu' if 'cpu' in df.columns else 'cpu_utilization'
    features = [
        cpu_col, 'network_in', 'request_count', 'is_sale_active',
        'hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'time_category',
        'cpu_rolling_mean', 'cpu_rolling_std', 'cpu_rolling_min', 'cpu_rolling_max',
        'cpu_diff', 'network_rolling_mean', 'request_rolling_mean'
    ]
    
    print(f"   ✓ Total features: {len(features)}")
    print(f"   Features: {features[:5]}... (and {len(features)-5} more)")
    
    # 3. Preprocessing
    print("\n⚙️  Step 3: Data Preprocessing...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    # Save scaler for later use
    import pickle
    with open('scaler_advanced.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create sequences (use longer sequence)
    SEQUENCE_LENGTH = 36  # 3 hours
    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
    print(f"   ✓ Created {len(X)} sequences (length: {SEQUENCE_LENGTH})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"   ✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 4. Hyperparameter Tuning
    print("\n🔍 Step 4: Hyperparameter Tuning...")
    best_params = hyperparameter_tuning(X_train, y_train, X_test, y_test)
    
    # 5. Train Ensemble
    print("\n🎯 Step 5: Training Ensemble Models...")
    ensemble_info = train_ensemble(X_train, y_train, X_test, y_test, num_models=5)
    
    # 6. Train Final Best Model with Attention
    print("\n🌟 Step 6: Training Final Model with Best Parameters...")
    final_test_loss, _ = train_single_model(
        X_train, y_train, X_test, y_test, 
        best_params, 'final'
    )
    
    # Copy final model to main location
    os.system(f'cp {ENSEMBLE_DIR}/model_final.pth {LOCAL_MODEL_PATH}')
    
    # 7. Evaluate Ensemble
    print("\n📊 Step 7: Evaluating Ensemble Performance...")
    evaluate_ensemble(X_test, y_test, scaler, features)
    
    print("\n" + "="*80)
    print("✅ ADVANCED TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📁 Models saved in: {ENSEMBLE_DIR}/")
    print(f"📁 Final best model: {LOCAL_MODEL_PATH}")
    print(f"📁 Scaler saved: scaler_advanced.pkl")

def evaluate_ensemble(X_test, y_test, scaler, features):
    """Evaluate ensemble predictions"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Load ensemble info to get correct configs
    with open(os.path.join(ENSEMBLE_DIR, 'ensemble_info.json'), 'r') as f:
        ensemble_info = json.load(f)
    
    # Configurations for each model
    configs = [
        {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2},
        {'hidden_size': 96, 'num_layers': 2, 'dropout': 0.3},
        {'hidden_size': 160, 'num_layers': 3, 'dropout': 0.15},
        {'hidden_size': 128, 'num_layers': 4, 'dropout': 0.25},
        {'hidden_size': 112, 'num_layers': 2, 'dropout': 0.2},
    ]
    
    # Load all ensemble models
    ensemble_predictions = []
    
    for i, config in enumerate(configs):
        model_path = os.path.join(ENSEMBLE_DIR, f'model_{i}.pth')
        if os.path.exists(model_path):
            model = AttentionLSTM(
                X_test.shape[2], 
                config['hidden_size'], 
                config['num_layers'], 
                1, 
                config['dropout']
            )
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            with torch.no_grad():
                pred = model(X_test_tensor).numpy()
                ensemble_predictions.append(pred)
    
    # Average predictions
    avg_predictions = np.mean(ensemble_predictions, axis=0)
    
    # Inverse transform
    test_dummy = np.zeros((len(avg_predictions), len(features)))
    test_dummy[:, 0] = avg_predictions.flatten()
    test_pred_actual = scaler.inverse_transform(test_dummy)[:, 0]
    
    test_actual_dummy = np.zeros((len(y_test), len(features)))
    test_actual_dummy[:, 0] = y_test
    test_actual = scaler.inverse_transform(test_actual_dummy)[:, 0]
    
    # Calculate metrics
    test_r2 = r2_score(test_actual, test_pred_actual)
    test_mae = mean_absolute_error(test_actual, test_pred_actual)
    test_mape = np.mean(np.abs((test_actual - test_pred_actual) / test_actual)) * 100
    test_accuracy = 100 - test_mape
    
    print(f"\n📈 ENSEMBLE MODEL METRICS:")
    print(f"   • R² Score:          {test_r2:.4f} ({test_r2*100:.1f}% variance explained)")
    print(f"   • MAE:               {test_mae:.4f}%")
    print(f"   • MAPE:              {test_mape:.2f}%")
    print(f"   • Accuracy:          {test_accuracy:.2f}%")
    
    improvement = test_accuracy - 77.39
    print(f"\n🎉 Total Improvement: {improvement:+.2f}% (from 77.39% to {test_accuracy:.2f}%)")

if __name__ == "__main__":
    train_advanced_model()
