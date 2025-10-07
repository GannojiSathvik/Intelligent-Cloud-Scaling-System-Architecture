# 📂 Project Structure

This document explains the organization of files in this repository.

## 📁 Directory Structure

```
intelligent-cloud-scaling-system/
│
├── 📂 Lambda Functions (AWS)
│   ├── DataCollectorFunction.py          # Collects CloudWatch metrics every 5 min
│   ├── ScalingLogicFunction.py           # ML-based predictive scaling logic
│   └── GetDashboardDataFunction.py       # Backend API for web dashboard
│
├── 📂 ML Training Scripts
│   ├── train_model.py                    # Original training script (77.39% accuracy)
│   ├── train_model_optimized.py          # Optimized version (78.87% accuracy) ⭐
│   ├── train_model_advanced.py           # All advanced techniques (ensemble)
│   ├── evaluate_model.py                 # Basic model evaluation
│   ├── evaluate_advanced.py              # Comprehensive evaluation
│   ├── predict.py                        # Make predictions with trained model
│   └── generate_synthetic_data.py        # Generate sample training data
│
├── 📂 Trained Models
│   ├── lstm_model.pth                    # Original LSTM model
│   ├── lstm_model_optimized.pth          # Best performing model ⭐ USE THIS
│   ├── lstm_model_advanced.pth           # Advanced features model
│   ├── lstm_model.h5                     # Placeholder (legacy)
│   ├── scaler_advanced.pkl               # Feature scaler for advanced model
│   └── ensemble_models/                  # 5 ensemble models
│       ├── model_0.pth to model_4.pth
│       └── ensemble_info.json
│
├── 📂 Web Dashboard
│   └── index.html                        # Real-time monitoring dashboard
│
├── 📂 Data Files
│   ├── multi_metric_data.csv             # Historical metrics dataset (4320 records)
│   ├── business_calendar.json            # Business events configuration
│   └── evaluation_results.json           # Model evaluation metrics
│
├── 📂 Documentation
│   ├── ACCURACY_IMPROVEMENT_GUIDE.md     # How we improved accuracy
│   ├── ADVANCED_TECHNIQUES_RESULTS.md    # Detailed ML techniques & results
│   └── deployment_guide.md               # AWS deployment instructions
│
├── 📄 README.md                          # Main project documentation (START HERE)
├── 📄 requirements.txt                    # Python dependencies
├── 📄 .gitignore                         # Git ignore rules
├── 📄 LICENSE                            # MIT License
└── 📄 STRUCTURE.md                       # This file
```

---

## 🎯 Quick Reference: Which File to Use?

### For Production Deployment:
- **Model**: `lstm_model_optimized.pth` (78.87% accuracy, R²=0.94)
- **Lambda Functions**: All 3 files in root directory
- **Dashboard**: `index.html`

### For Training:
- **Recommended**: `train_model_optimized.py`
- **Advanced**: `train_model_advanced.py` (ensemble + all techniques)

### For Evaluation:
- **Quick check**: `evaluate_model.py`
- **Comprehensive**: `evaluate_advanced.py`

### For Predictions:
- **Script**: `predict.py`
- **Data**: Uses last 12 rows from `multi_metric_data.csv`

---

## 📊 File Sizes & Purpose

| File | Size | Purpose | Priority |
|------|------|---------|----------|
| `lstm_model_optimized.pth` | 1.3 MB | **Production model** | 🔴 Critical |
| `multi_metric_data.csv` | 350 KB | Training data | 🔴 Critical |
| `index.html` | 18 KB | Web dashboard | 🟡 Important |
| `train_model_optimized.py` | 9 KB | Training script | 🟡 Important |
| `DataCollectorFunction.py` | 4 KB | Data collection | 🔴 Critical |
| `ScalingLogicFunction.py` | 5 KB | Scaling logic | 🔴 Critical |
| `GetDashboardDataFunction.py` | 6 KB | Dashboard API | 🟡 Important |

---

## 🚀 Workflow: How Files Interact

### 1. Data Collection (Every 5 minutes)
```
CloudWatch → DataCollectorFunction.py → multi_metric_data.csv (S3)
                                      ↘ business_calendar.json
```

### 2. Model Training (Daily/On-demand)
```
multi_metric_data.csv → train_model_optimized.py → lstm_model_optimized.pth
```

### 3. Predictive Scaling (Every 5 minutes)
```
multi_metric_data.csv → ScalingLogicFunction.py → Auto Scaling Group
lstm_model_optimized.pth ↗
```

### 4. Dashboard (Real-time)
```
S3 Data → GetDashboardDataFunction.py → API Gateway → index.html
```

---

## 📝 File Descriptions

### Lambda Functions

#### `DataCollectorFunction.py`
- **Purpose**: Collect metrics from CloudWatch
- **Trigger**: CloudWatch Events (every 5 minutes)
- **Input**: CloudWatch metrics API
- **Output**: Appends to `multi_metric_data.csv` in S3
- **Key Functions**:
  - `lambda_handler()` - Main entry point
  - `get_cloudwatch_metrics()` - Fetch CPU, network, requests
  - `get_business_context()` - Read sale status

#### `ScalingLogicFunction.py`
- **Purpose**: Predict load and scale infrastructure
- **Trigger**: CloudWatch Events (every 5 minutes)
- **Input**: Historical data + trained model
- **Output**: Scaling decisions (add/remove instances)
- **Key Functions**:
  - `lambda_handler()` - Main entry point
  - `predict_load()` - ML prediction
  - `make_scaling_decision()` - Scale up/down logic

#### `GetDashboardDataFunction.py`
- **Purpose**: Provide data for web dashboard
- **Trigger**: API Gateway (HTTP GET)
- **Input**: S3 data + ASG status
- **Output**: JSON response
- **Key Functions**:
  - `lambda_handler()` - Main entry point
  - `get_current_server_count()` - ASG capacity
  - `get_historical_metrics()` - Last 100 records

### Training Scripts

#### `train_model_optimized.py` ⭐ RECOMMENDED
- **Purpose**: Train optimized LSTM model
- **Features**: 4 (CPU, network, requests, sale status)
- **Sequence Length**: 24 timesteps (2 hours)
- **Architecture**: 3 layers, 128 hidden units
- **Output**: `lstm_model_optimized.pth` (78.87% accuracy)

#### `train_model_advanced.py`
- **Purpose**: Train with ALL advanced techniques
- **Features**: 16 (engineered temporal features)
- **Sequence Length**: 36 timesteps (3 hours)
- **Techniques**: Attention, ensemble, hyperparameter tuning
- **Output**: Multiple models in `ensemble_models/`

#### `evaluate_advanced.py`
- **Purpose**: Comprehensive model evaluation
- **Input**: All trained models
- **Output**: `evaluation_results.json`
- **Metrics**: Accuracy, R², MAE, MAPE for each model

### Web Dashboard

#### `index.html`
- **Purpose**: Real-time monitoring interface
- **Features**:
  - 4 interactive charts (CPU, server count, network, requests)
  - Current metrics display
  - Auto-refresh (30 seconds)
  - Responsive design
- **Dependencies**: Chart.js (loaded via CDN)
- **API**: Fetches from `GetDashboardDataFunction`

---

## 🔄 Data Flow

```
1. COLLECTION
   CloudWatch Metrics → DataCollectorFunction
                              ↓
                          S3 Bucket
                              ↓
                     multi_metric_data.csv

2. TRAINING (Periodic)
   multi_metric_data.csv → train_model_optimized.py
                                    ↓
                           lstm_model_optimized.pth

3. PREDICTION (Real-time)
   Last 24 rows of data → ScalingLogicFunction
   + lstm_model_optimized.pth
                              ↓
                    Scaling Decision
                              ↓
                     Auto Scaling Group

4. VISUALIZATION
   S3 Data → GetDashboardDataFunction → API Gateway
                                              ↓
                                         Web Browser
                                              ↓
                                         index.html
```

---

## 🎨 File Organization Best Practices

### ✅ Current Organization (As-Is)
All files are in the root directory for simplicity.

### 🚀 Recommended Organization (For Production)
```
intelligent-cloud-scaling-system/
├── lambda/                    # AWS Lambda functions
├── models/                    # Trained models
├── scripts/                   # Training/evaluation scripts
├── frontend/                  # Web dashboard
├── data/                      # Sample data
└── docs/                      # Documentation
```

To reorganize (optional):
```bash
mkdir -p lambda models scripts frontend data docs
mv *Function.py lambda/
mv *.pth *.pkl ensemble_models models/
mv train_*.py evaluate_*.py predict.py generate_*.py scripts/
mv index.html frontend/
mv *.csv *.json data/
mv *.md docs/ (except README.md)
```

---

## 📌 Important Notes

1. **Production Model**: Always use `lstm_model_optimized.pth` (best performance)
2. **Data Updates**: `multi_metric_data.csv` grows continuously (archive old data periodically)
3. **Business Calendar**: Update `business_calendar.json` for sales/events
4. **Model Retraining**: Retrain weekly/monthly as data patterns change
5. **Dashboard Config**: Update API endpoint in `index.html` after deployment

---

## 🔗 Related Documentation

- [README.md](README.md) - Project overview
- [ACCURACY_IMPROVEMENT_GUIDE.md](ACCURACY_IMPROVEMENT_GUIDE.md) - ML improvements
- [ADVANCED_TECHNIQUES_RESULTS.md](ADVANCED_TECHNIQUES_RESULTS.md) - Detailed results
- [deployment_guide.md](deployment_guide.md) - AWS setup

---

**Last Updated**: 2025-10-06
