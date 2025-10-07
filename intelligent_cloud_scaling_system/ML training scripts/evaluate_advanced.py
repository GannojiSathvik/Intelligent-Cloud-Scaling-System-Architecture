"""
Evaluation script for the advanced models with all techniques
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
import os

# Import from training script
from train_model_advanced import (
    add_temporal_features, 
    create_sequences, 
    AttentionLSTM,
    ENSEMBLE_DIR
)

def evaluate_all_models():
    """Evaluate and compare all trained models"""
    
    print("="*80)
    print("🔍 COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    
    # 1. Load and prepare data
    print("\n📊 Loading Data...")
    df = pd.read_csv('multi_metric_data.csv', parse_dates=['timestamp'])
    df = df.sort_values('timestamp')
    df = add_temporal_features(df)
    
    cpu_col = 'cpu' if 'cpu' in df.columns else 'cpu_utilization'
    features = [
        cpu_col, 'network_in', 'request_count', 'is_sale_active',
        'hour', 'day_of_week', 'is_weekend', 'is_business_hours', 'time_category',
        'cpu_rolling_mean', 'cpu_rolling_std', 'cpu_rolling_min', 'cpu_rolling_max',
        'cpu_diff', 'network_rolling_mean', 'request_rolling_mean'
    ]
    
    with open('scaler_advanced.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    scaled_data = scaler.transform(df[features])
    X, y = create_sequences(scaled_data, 36)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"   ✓ Test samples: {len(X_test)}")
    print(f"   ✓ Features: {len(features)}")
    
    # 2. Evaluate individual ensemble models
    print("\n" + "="*80)
    print("📊 INDIVIDUAL MODEL PERFORMANCE")
    print("="*80)
    
    configs = [
        {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.2, 'name': 'Model 1 (H128-L3)'},
        {'hidden_size': 96, 'num_layers': 2, 'dropout': 0.3, 'name': 'Model 2 (H96-L2)'},
        {'hidden_size': 160, 'num_layers': 3, 'dropout': 0.15, 'name': 'Model 3 (H160-L3)'},
        {'hidden_size': 128, 'num_layers': 4, 'dropout': 0.25, 'name': 'Model 4 (H128-L4)'},
        {'hidden_size': 112, 'num_layers': 2, 'dropout': 0.2, 'name': 'Model 5 (H112-L2)'},
    ]
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    ensemble_predictions = []
    individual_metrics = []
    
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
            
            # Inverse transform
            test_dummy = np.zeros((len(pred), len(features)))
            test_dummy[:, 0] = pred.flatten()
            test_pred_actual = scaler.inverse_transform(test_dummy)[:, 0]
            
            test_actual_dummy = np.zeros((len(y_test), len(features)))
            test_actual_dummy[:, 0] = y_test
            test_actual = scaler.inverse_transform(test_actual_dummy)[:, 0]
            
            # Metrics
            r2 = r2_score(test_actual, test_pred_actual)
            mae = mean_absolute_error(test_actual, test_pred_actual)
            mape = np.mean(np.abs((test_actual - test_pred_actual) / test_actual)) * 100
            accuracy = 100 - mape
            
            print(f"\n{config['name']}:")
            print(f"   • R² Score:   {r2:.4f}")
            print(f"   • MAE:        {mae:.4f}%")
            print(f"   • Accuracy:   {accuracy:.2f}%")
            
            ensemble_predictions.append(pred)
            individual_metrics.append({
                'name': config['name'],
                'r2': r2,
                'mae': mae,
                'accuracy': accuracy
            })
    
    # 3. Evaluate ensemble
    print("\n" + "="*80)
    print("🎯 ENSEMBLE MODEL PERFORMANCE")
    print("="*80)
    
    avg_predictions = np.mean(ensemble_predictions, axis=0)
    
    test_dummy = np.zeros((len(avg_predictions), len(features)))
    test_dummy[:, 0] = avg_predictions.flatten()
    ensemble_pred_actual = scaler.inverse_transform(test_dummy)[:, 0]
    
    test_actual_dummy = np.zeros((len(y_test), len(features)))
    test_actual_dummy[:, 0] = y_test
    test_actual = scaler.inverse_transform(test_actual_dummy)[:, 0]
    
    ensemble_r2 = r2_score(test_actual, ensemble_pred_actual)
    ensemble_mae = mean_absolute_error(test_actual, ensemble_pred_actual)
    ensemble_mape = np.mean(np.abs((test_actual - ensemble_pred_actual) / test_actual)) * 100
    ensemble_accuracy = 100 - ensemble_mape
    
    print(f"\n🌟 Ensemble (Average of 5 models):")
    print(f"   • R² Score:   {ensemble_r2:.4f} ({ensemble_r2*100:.1f}% variance)")
    print(f"   • MAE:        {ensemble_mae:.4f}%")
    print(f"   • MAPE:       {ensemble_mape:.2f}%")
    print(f"   • Accuracy:   {ensemble_accuracy:.2f}%")
    
    # 4. Compare with baseline
    print("\n" + "="*80)
    print("📊 COMPARISON WITH BASELINE")
    print("="*80)
    
    print(f"\n{'Model':<30} {'Accuracy':<12} {'R² Score':<12} {'MAE':<10}")
    print("-" * 80)
    print(f"{'Original (77.39%)':<30} {'77.39%':<12} {'0.9340':<12} {'4.37%':<10}")
    print(f"{'Optimized (78.87%)':<30} {'78.87%':<12} {'0.9400':<12} {'4.13%':<10}")
    
    for m in individual_metrics:
        acc_str = f"{m['accuracy']:.2f}%"
        r2_str = f"{m['r2']:.4f}"
        mae_str = f"{m['mae']:.2f}%"
        print(f"{m['name']:<30} {acc_str:<12} {r2_str:<12} {mae_str:<10}")
    
    ensemble_acc_str = f"{ensemble_accuracy:.2f}%"
    ensemble_r2_str = f"{ensemble_r2:.4f}"
    ensemble_mae_str = f"{ensemble_mae:.2f}%"
    print(f"{'Ensemble (Advanced)':<30} {ensemble_acc_str:<12} {ensemble_r2_str:<12} {ensemble_mae_str:<10}")
    
    # 5. Summary
    print("\n" + "="*80)
    print("🎉 SUMMARY")
    print("="*80)
    
    improvements = [
        ("Original → Optimized", 78.87 - 77.39),
        ("Original → Best Individual", max(m['accuracy'] for m in individual_metrics) - 77.39),
        ("Original → Ensemble", ensemble_accuracy - 77.39)
    ]
    
    print(f"\n📈 Accuracy Improvements:")
    for desc, imp in improvements:
        print(f"   • {desc:<30} {imp:+.2f}%")
    
    best_model = max(individual_metrics, key=lambda x: x['accuracy'])
    print(f"\n🏆 Best Individual Model: {best_model['name']} ({best_model['accuracy']:.2f}%)")
    print(f"🏆 Best Overall: Ensemble ({ensemble_accuracy:.2f}%)")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    
    # Save results
    results = {
        'individual_models': individual_metrics,
        'ensemble': {
            'accuracy': ensemble_accuracy,
            'r2': ensemble_r2,
            'mae': ensemble_mae,
            'mape': ensemble_mape
        }
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: evaluation_results.json")

if __name__ == "__main__":
    evaluate_all_models()
