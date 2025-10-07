# 🎓 Professor Demonstration Guide
## Intelligent Cloud Scaling System with Machine Learning

**Student:** Gannoji Sathvik  
**AWS Account:** 780139019966  
**Date:** October 6, 2025  
**Project:** Intelligent Cloud Scaling System Architecture

---

## 📊 Executive Summary

This project implements an **intelligent, predictive cloud scaling system** using machine learning (LSTM neural networks) to forecast CPU utilization and proactively scale AWS Auto Scaling Groups **before** demand spikes occur.

### Key Achievements

✅ **Model Accuracy: 78.51%** (Best Individual Model)  
✅ **R² Score: 0.9328** (93.3% variance explained)  
✅ **Mean Absolute Error: 4.36%**  
✅ **Ensemble of 5 Models** for robust predictions  
✅ **Fully Automated AWS Deployment**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AWS CLOUD ENVIRONMENT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   EC2        │      │   EC2        │      │   EC2        │  │
│  │  Instance    │◄─────┤  Instance    │◄─────┤  Instance    │  │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘  │
│         │                     │                     │            │
│         └─────────────────────┴─────────────────────┘            │
│                              │                                   │
│                   ┌──────────▼──────────┐                        │
│                   │  Auto Scaling Group │                        │
│                   │  (intelligent-      │                        │
│                   │   scaling-demo)     │                        │
│                   └──────────┬──────────┘                        │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐              │
│         │                    │                    │              │
│  ┌──────▼────────┐   ┌──────▼────────┐   ┌──────▼────────┐     │
│  │  CloudWatch   │   │   Lambda      │   │   Lambda      │     │
│  │   Metrics     │──►│ DataCollector │   │ ScalingLogic  │     │
│  └───────────────┘   └───────┬───────┘   └───────┬───────┘     │
│                              │                    │              │
│                              ▼                    ▼              │
│                      ┌───────────────────────────────┐           │
│                      │         S3 Bucket             │           │
│                      │  my-intelligent-scaling-      │           │
│                      │      data-bucket              │           │
│                      │                               │           │
│                      │  • multi_metric_data.csv      │           │
│                      │  • models/lstm_model.pth      │           │
│                      │  • business_calendar.json     │           │
│                      └───────────────────────────────┘           │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              EventBridge (CloudWatch Events)               │  │
│  │  Triggers every 5 minutes: Data Collection & Prediction    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Machine Learning Model

### Model Architecture: LSTM with Attention Mechanism

**Features Used (16 total):**
- CPU Utilization (current and rolling statistics)
- Network In/Out
- Request Count
- Business Context (sale events)
- Temporal Features (hour, day of week, business hours)
- Rate of Change Indicators

**Model Specifications:**
- **Type:** Attention-based LSTM (Long Short-Term Memory)
- **Sequence Length:** 36 time steps (3 hours of data)
- **Hidden Layers:** 3-4 layers with 96-160 neurons
- **Dropout:** 0.15-0.3 for regularization
- **Framework:** PyTorch

### Training Results

| Model | Accuracy | R² Score | MAE |
|-------|----------|----------|-----|
| **Model 3 (Best)** | **78.51%** | **0.9317** | **4.34%** |
| Model 1 | 78.30% | 0.9303 | 4.34% |
| Model 2 | 77.72% | 0.9364 | 4.34% |
| Ensemble | 77.86% | 0.9328 | 4.36% |

**Improvement over Baseline:** +1.12% accuracy

---

## 🚀 AWS Components

### 1. S3 Bucket: `my-intelligent-scaling-data-bucket`
**Purpose:** Central data repository
- Stores historical metrics (multi_metric_data.csv)
- Hosts trained ML model (lstm_model.pth)
- Maintains business calendar for context

### 2. Lambda Function: `DataCollectorFunction`
**Purpose:** Automated data collection
- **Runtime:** Python 3.9
- **Trigger:** Every 5 minutes (EventBridge)
- **Actions:**
  - Fetches CloudWatch metrics (CPU, Network, etc.)
  - Reads business context from S3
  - Appends new data to CSV in S3

### 3. Lambda Function: `ScalingLogicFunction`
**Purpose:** Predictive scaling decisions
- **Runtime:** Python 3.9 + PyTorch Layer
- **Trigger:** Every 5 minutes (EventBridge)
- **Actions:**
  - Downloads trained LSTM model from S3
  - Fetches last 3 hours of metrics
  - Predicts CPU utilization for next 5 minutes
  - Adjusts Auto Scaling Group capacity proactively

**Scaling Thresholds:**
- **Scale Up:** Predicted CPU > 70%
- **Scale Down:** Predicted CPU < 35%

### 4. Auto Scaling Group: `intelligent-scaling-demo-sathvik`
**Purpose:** Managed EC2 instance fleet
- Automatically adjusts capacity based on ML predictions
- Maintains optimal resource allocation

### 5. EventBridge Rules
**Purpose:** Automated scheduling
- **DataCollectorSchedule:** Triggers data collection every 5 minutes
- **ScalingLogicSchedule:** Triggers prediction and scaling every 5 minutes

---

## 📈 Demonstration Steps

### Step 1: Verify System Status

```bash
# Run verification script
./verify_aws.sh
```

**Expected Output:**
- ✅ AWS credentials configured
- ✅ S3 bucket exists with data
- ✅ Lambda functions deployed
- ✅ Auto Scaling Group operational
- ✅ EventBridge rules active

### Step 2: Check Data Collection

```bash
# View recent data in S3
aws s3 cp s3://my-intelligent-scaling-data-bucket/multi_metric_data.csv - | tail -20

# Check data file size
aws s3 ls s3://my-intelligent-scaling-data-bucket/ --human-readable
```

**What to Show:**
- CSV file growing over time
- Timestamps showing 5-minute intervals
- Metrics: CPU, Network, Request Count, Business Context

### Step 3: Monitor Lambda Execution

```bash
# View DataCollector logs
aws logs tail /aws/lambda/DataCollectorFunction --follow

# View ScalingLogic logs
aws logs tail /aws/lambda/ScalingLogicFunction --follow
```

**What to Show:**
- Successful executions every 5 minutes
- CPU predictions being made
- Scaling decisions (scale up/down/no action)

### Step 4: Observe Auto Scaling Activity

```bash
# Check current ASG status
aws autoscaling describe-auto-scaling-groups \
  --auto-scaling-group-names intelligent-scaling-demo-sathvik \
  --query 'AutoScalingGroups[0].[DesiredCapacity,MinSize,MaxSize]'

# View recent scaling activities
aws autoscaling describe-scaling-activities \
  --auto-scaling-group-name intelligent-scaling-demo-sathvik \
  --max-records 10
```

**What to Show:**
- Current instance count
- Scaling history with timestamps
- Correlation between predictions and actions

### Step 5: Model Performance Visualization

```bash
# Show evaluation results
cat evaluation_results.json | python -m json.tool
```

**What to Show:**
- Individual model accuracies
- Ensemble performance
- Comparison with baseline

---

## 🎯 Key Differentiators

### 1. **Proactive vs Reactive**
- **Traditional:** Reacts after CPU threshold is breached
- **This System:** Predicts and acts before threshold is reached

### 2. **Context-Aware**
- Incorporates business events (sales, promotions)
- Considers temporal patterns (time of day, day of week)
- Uses rolling statistics for trend detection

### 3. **Ensemble Learning**
- Multiple models for robust predictions
- Reduces overfitting and improves generalization

### 4. **Fully Automated**
- No manual intervention required
- Self-learning from historical data
- Continuous improvement as more data is collected

---

## 📊 Performance Metrics

### Model Accuracy
- **Best Individual Model:** 78.51%
- **Ensemble Model:** 77.86%
- **R² Score:** 0.9328 (93.3% variance explained)

### Operational Metrics
- **Prediction Frequency:** Every 5 minutes
- **Data Collection:** Automated every 5 minutes
- **Response Time:** < 60 seconds from prediction to scaling action

### Cost Efficiency
- **Lambda Executions:** ~17,280 per month (2 functions × 12 per hour × 720 hours)
- **Estimated Cost:** < $5/month for Lambda + S3
- **Savings:** Reduced over-provisioning and improved resource utilization

---

## 🔧 Technical Implementation

### Local Development
```bash
# Train models locally
python train_model_advanced.py

# Evaluate models
python evaluate_advanced.py

# Test predictions
python predict.py
```

### AWS Deployment
```bash
# Complete setup (one command)
chmod +x setup_aws_complete.sh
./setup_aws_complete.sh

# Verify deployment
./verify_aws.sh
```

---

## 📚 Project Files

### Core Files
- `train_model_advanced.py` - Advanced LSTM training with attention
- `evaluate_advanced.py` - Model evaluation and metrics
- `ScalingLogicFunction.py` - Lambda function for predictions
- `DataCollectorFunction.py` - Lambda function for data collection

### Configuration
- `requirements.txt` - Python dependencies
- `business_calendar.json` - Business context data

### Documentation
- `README.md` - Project overview
- `STRUCTURE.md` - Repository structure
- `QUICKSTART.md` - Quick start guide
- `PROFESSOR_DEMONSTRATION.md` - This file

### Trained Models
- `lstm_model_advanced.pth` - Best performing model
- `ensemble_models/` - All 5 ensemble models
- `scaler_advanced.pkl` - Feature scaler

---

## 🎓 Learning Outcomes

1. **Machine Learning:** LSTM networks, attention mechanisms, ensemble methods
2. **Cloud Computing:** AWS Lambda, S3, Auto Scaling, EventBridge
3. **DevOps:** Infrastructure as Code, automated deployment
4. **Data Engineering:** Time series processing, feature engineering
5. **System Design:** Scalable, fault-tolerant architecture

---

## 🔗 Quick Access Links

### AWS Console
- **S3 Bucket:** https://s3.console.aws.amazon.com/s3/buckets/my-intelligent-scaling-data-bucket
- **Lambda Functions:** https://console.aws.amazon.com/lambda/home#/functions
- **Auto Scaling:** https://console.aws.amazon.com/ec2/autoscaling/home#AutoScalingGroups
- **EventBridge:** https://console.aws.amazon.com/events/home#/rules
- **CloudWatch Logs:** https://console.aws.amazon.com/cloudwatch/home#logsV2

### Monitoring
```bash
# Real-time logs
aws logs tail /aws/lambda/ScalingLogicFunction --follow

# System status
./verify_aws.sh
```

---

## ✅ Demonstration Checklist

Before presenting to professor:

- [ ] Run `./verify_aws.sh` - All components operational
- [ ] Check S3 bucket has data and model files
- [ ] Verify Lambda functions are executing (check CloudWatch Logs)
- [ ] Confirm Auto Scaling Group exists and has instances
- [ ] Review evaluation_results.json for model metrics
- [ ] Prepare to show real-time predictions in CloudWatch Logs
- [ ] Have architecture diagram ready
- [ ] Explain proactive vs reactive scaling benefits

---

## 🎉 Conclusion

This project demonstrates a **production-ready, intelligent cloud scaling system** that:
- Achieves **78.51% prediction accuracy**
- Operates **fully autonomously** on AWS
- Provides **proactive scaling** before demand spikes
- Incorporates **business context** for smarter decisions
- Uses **state-of-the-art ML techniques** (LSTM + Attention)

**Status:** ✅ **READY FOR DEMONSTRATION**

---

**For Questions or Issues:**
- Check CloudWatch Logs for Lambda execution details
- Run `./verify_aws.sh` for system status
- Review `evaluation_results.json` for model performance
