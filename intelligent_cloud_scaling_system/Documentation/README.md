# 🚀 Intelligent Cloud Scaling System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![AWS](https://img.shields.io/badge/AWS-Lambda%20%7C%20S3%20%7C%20CloudWatch-orange.svg)](https://aws.amazon.com/)

> **A sophisticated, ML-powered predictive auto-scaling system for AWS cloud infrastructure using LSTM neural networks and real-time business context awareness.**

![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-78.87%25-brightgreen)
![R² Score](https://img.shields.io/badge/R²%20Score-0.94-blue)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [AWS Deployment](#aws-deployment)
- [Web Dashboard](#web-dashboard)
- [Model Training](#model-training)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements an **intelligent, predictive auto-scaling system** that goes beyond traditional reactive cloud scaling. Using advanced **LSTM (Long Short-Term Memory)** neural networks and **PyTorch**, the system predicts future server load and scales infrastructure **proactively** before demand spikes occur.

### How It Works: Three-Stage Pipeline

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Data      │─────▶│   Machine    │─────▶│   Predictive    │
│ Collection  │      │   Learning   │      │    Scaling      │
│  (Every 5m) │      │   Training   │      │   (Every 5m)    │
└─────────────┘      └──────────────┘      └─────────────────┘
     ↓                      ↓                       ↓
 CloudWatch            LSTM Model              Auto Scaling
 + Business            (PyTorch)                   Group
  Context
```

### 1️⃣ **Data Collection** (`lambda_functions/DataCollectorFunction.py`)
- **Every 5 minutes**: Collects real-time metrics from AWS CloudWatch
- **Metrics gathered**: CPU utilization, network traffic, request count
- **Business context**: Reads `business_calendar.json` for events (sales, promotions)
- **Storage**: Appends to historical dataset in S3 (`multi_metric_data.csv`)

### 2️⃣ **Model Training** (`scripts/train_model_optimized.py`)
- **Frequency**: Daily or on-demand
- **Algorithm**: LSTM neural network with attention mechanism
- **Features**: 16 engineered features including temporal patterns
- **Output**: Trained model achieving **78.87% accuracy** and **R² = 0.94**
- **Framework**: PyTorch with advanced optimizations

### 3️⃣ **Predictive Scaling** (`lambda_functions/ScalingLogicFunction.py`)
- **Every 5 minutes**: Makes prediction for next 5-minute window
- **Decision logic**:
  - **Predicted CPU > 70%** → Scale UP (add instance)
  - **Predicted CPU < 35%** → Scale DOWN (remove instance)
- **Proactive**: Scales **before** demand hits, not after

---

## ✨ Key Features

### 🔮 **Predictive, Not Reactive**
- **Traditional**: Waits for high CPU, then reacts (lag = poor performance)
- **Our System**: Forecasts load and scales **in advance** (prevents issues)

### 📊 **Multi-Variate Analysis**
- **Traditional**: Single metric (CPU only)
- **Our System**: 16 features including:
  - Server metrics (CPU, network, requests)
  - Temporal features (hour, day, weekend)
  - Rolling statistics (mean, std, min, max)
  - Business context (sales events)

### 🧠 **Context-Aware Intelligence**
- **Traditional**: Blind to business events
- **Our System**: Understands business context
  - Recognizes sale days vs. normal days
  - Learns different traffic patterns
  - Makes aggressive scaling for planned events

### 🎨 **Real-Time Dashboard**
- Beautiful web UI with live metrics
- Interactive charts (CPU, network, requests, server count)
- Auto-refreshes every 30 seconds
- Responsive design

### 🏆 **Advanced ML Techniques**
- ✅ Attention mechanism LSTM
- ✅ Feature engineering (16 features)
- ✅ Ensemble models (5 diverse architectures)
- ✅ Hyperparameter tuning (grid search)
- ✅ Early stopping & learning rate scheduling

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Cloud                             │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │              │    │              │    │              │  │
│  │  CloudWatch  │───▶│   Lambda     │───▶│     S3       │  │
│  │   Metrics    │    │ (Collector)  │    │   Bucket     │  │
│  │              │    │              │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                 │            │
│                                                 ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │              │    │              │    │              │  │
│  │ Auto Scaling │◀───│   Lambda     │◀───│  ML Model    │  │
│  │    Group     │    │  (Scaler)    │    │  (PyTorch)   │  │
│  │              │    │              │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          API Gateway + Dashboard Lambda              │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
└────────────────────────────┼─────────────────────────────────┘
                             ▼
                    ┌─────────────────┐
                    │  Web Dashboard  │
                    │   (index.html)  │
                    └─────────────────┘
```

---

## 📈 Model Performance

### 🎯 **Final Results**

| Model Version | Accuracy | R² Score | MAE | Features |
|--------------|----------|----------|-----|----------|
| **Original** | 77.39% | 0.9340 | 4.37% | 4 basic features |
| **Optimized** | **78.87%** | **0.9400** | **4.13%** | Architecture improvements |
| **Advanced (Best)** | 78.51% | 0.9317 | 4.34% | 16 features + attention |
| **Ensemble** | 77.86% | 0.9328 | 4.36% | 5 model average |

### 🔧 **Model Architecture (Optimized)**
```python
- Input: 24 timesteps × 4 features (2 hours of history)
- LSTM Layers: 3 layers, 128 hidden units
- Dropout: 0.2 (20% regularization)
- Dense Layers: 128 → 64 → 1
- Optimizer: Adam (lr=0.0005)
- Loss: MSE with gradient clipping
```

### 📊 **Performance Metrics**
- **R² Score**: 0.94 (explains 94% of variance)
- **Mean Absolute Error**: ±4.13% CPU utilization
- **MAPE**: 21.13%
- **Training Time**: ~45 seconds on CPU
- **Inference Time**: <100ms

---

## 📁 Project Structure

```
intelligent-cloud-scaling-system/
│
├── 📂 lambda_functions/          # AWS Lambda functions
│   ├── DataCollectorFunction.py  # Collects metrics every 5 min
│   ├── ScalingLogicFunction.py   # ML-based scaling logic
│   └── GetDashboardDataFunction.py # Dashboard API backend
│
├── 📂 scripts/                    # Training & evaluation
│   ├── train_model.py            # Original training (77.39%)
│   ├── train_model_optimized.py  # Optimized (78.87%)
│   ├── train_model_advanced.py   # All techniques (ensemble)
│   ├── evaluate_model.py         # Basic evaluation
│   ├── evaluate_advanced.py      # Comprehensive evaluation
│   ├── predict.py                # Prediction script
│   └── generate_synthetic_data.py # Data generation
│
├── 📂 models/                     # Trained models
│   ├── lstm_model.pth            # Original model
│   ├── lstm_model_optimized.pth  # Best model (78.87%)
│   ├── lstm_model_advanced.pth   # Advanced features
│   ├── scaler_advanced.pkl       # Feature scaler
│   └── ensemble_models/          # 5 ensemble models
│
├── 📂 frontend/                   # Web dashboard
│   └── index.html                # Real-time dashboard UI
│
├── 📂 data/                       # Sample data
│   ├── multi_metric_data.csv     # Historical metrics
│   ├── business_calendar.json    # Business events
│   └── evaluation_results.json   # Model metrics
│
├── 📂 docs/                       # Documentation
│   ├── ACCURACY_IMPROVEMENT_GUIDE.md
│   ├── ADVANCED_TECHNIQUES_RESULTS.md
│   └── deployment_guide.md
│
├── 📄 README.md                   # This file
├── 📄 requirements.txt            # Python dependencies
└── 📄 .gitignore                  # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- AWS Account with CLI configured
- PyTorch 1.9+

### Quick Start

```bash
# Clone repository
git clone https://github.com/GannojiSathvik/Intelligent-Cloud-Scaling-System-Architecture.git
cd Intelligent-Cloud-Scaling-System-Architecture

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate sample data (optional)
python scripts/generate_synthetic_data.py

# Train model
python scripts/train_model_optimized.py

# Evaluate model
python scripts/evaluate_model.py
```

---

## 💻 Usage

### 1. Train the Model

```bash
# Basic training
python scripts/train_model.py

# Optimized training (recommended)
python scripts/train_model_optimized.py

# Advanced training (all techniques)
python scripts/train_model_advanced.py
```

### 2. Make Predictions

```python
from scripts.predict import run_prediction

# Predict CPU for next 5 minutes
run_prediction()
```

### 3. Evaluate Performance

```bash
# Basic evaluation
python scripts/evaluate_model.py

# Comprehensive evaluation
python scripts/evaluate_advanced.py
```

---

## ☁️ AWS Deployment

### Step 1: Deploy Lambda Functions

```bash
# Package Lambda functions
cd lambda_functions
zip -r data_collector.zip DataCollectorFunction.py
zip -r scaling_logic.zip ScalingLogicFunction.py
zip -r dashboard_api.zip GetDashboardDataFunction.py

# Upload to AWS Lambda (using AWS CLI)
aws lambda create-function \
  --function-name DataCollector \
  --runtime python3.9 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-role \
  --handler DataCollectorFunction.lambda_handler \
  --zip-file fileb://data_collector.zip
```

### Step 2: Configure S3 Bucket

```bash
# Create S3 bucket
aws s3 mb s3://intelligent-scaling-demo-YOUR-NAME

# Upload initial files
aws s3 cp data/business_calendar.json s3://intelligent-scaling-demo-YOUR-NAME/
aws s3 cp models/lstm_model_optimized.pth s3://intelligent-scaling-demo-YOUR-NAME/models/
```

### Step 3: Set Up CloudWatch Events

```bash
# Schedule DataCollector every 5 minutes
aws events put-rule \
  --name DataCollectionSchedule \
  --schedule-expression "rate(5 minutes)"

# Schedule ScalingLogic every 5 minutes
aws events put-rule \
  --name ScalingSchedule \
  --schedule-expression "rate(5 minutes)"
```

### Step 4: Deploy Dashboard

```bash
# Create API Gateway endpoint
aws apigateway create-rest-api --name DashboardAPI

# Update index.html with API endpoint
# Upload to S3 with static website hosting enabled
aws s3 cp frontend/index.html s3://your-dashboard-bucket/ --acl public-read
```

---

## 📊 Web Dashboard

### Features
- **Real-time metrics** display
- **Interactive charts** (CPU, network, requests, servers)
- **Auto-refresh** every 30 seconds
- **Responsive design** for mobile/desktop
- **Beautiful gradients** and animations

### Access Dashboard
1. Deploy `frontend/index.html` to S3 static hosting
2. Configure API Gateway endpoint in HTML
3. Access via: `http://your-bucket.s3-website-region.amazonaws.com`

![Dashboard Preview](https://via.placeholder.com/800x400?text=Dashboard+Preview)

---

## 🧠 Model Training

### Training Process

```python
# 1. Load historical data
df = pd.read_csv('data/multi_metric_data.csv')

# 2. Feature engineering (16 features)
df = add_temporal_features(df)

# 3. Create sequences (36 timesteps)
X, y = create_sequences(scaled_data, SEQUENCE_LENGTH=36)

# 4. Train LSTM model
model = ImprovedLSTMModel(...)
model.fit(X_train, y_train, epochs=50)

# 5. Evaluate
accuracy = evaluate(model, X_test, y_test)
```

### Advanced Techniques Used

1. **Feature Engineering**
   - Temporal: hour, day, weekend, business hours
   - Rolling stats: mean, std, min, max
   - Rate of change

2. **Attention Mechanism**
   - Focus on important timesteps
   - Better pattern recognition

3. **Ensemble Models**
   - 5 diverse architectures
   - Averaged predictions

4. **Hyperparameter Tuning**
   - Grid search (10 configurations)
   - Best params auto-selected

---

## 📚 Documentation

- **[Accuracy Improvement Guide](docs/ACCURACY_IMPROVEMENT_GUIDE.md)** - How we improved from 77% to 79%
- **[Advanced Techniques Results](docs/ADVANCED_TECHNIQUES_RESULTS.md)** - All ML techniques applied
- **[Deployment Guide](docs/deployment_guide.md)** - Step-by-step AWS setup

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Gannoji Sathvik**
- GitHub: [@GannojiSathvik](https://github.com/GannojiSathvik)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/gannojisathvik)

---

## 🙏 Acknowledgments

- AWS for cloud infrastructure
- PyTorch team for the ML framework
- scikit-learn for preprocessing tools
- Chart.js for visualization

---

## 📞 Support

For questions or support:
- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/GannojiSathvik/Intelligent-Cloud-Scaling-System-Architecture/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/GannojiSathvik/Intelligent-Cloud-Scaling-System-Architecture/discussions)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ and ☕ by Sathvik

</div>
