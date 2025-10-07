# ⚡ Quick Start Guide

Get the Intelligent Cloud Scaling System up and running in **5 minutes**!

---

## 🎯 What You'll Build

A production-ready ML-powered auto-scaling system with:
- ✅ 78.87% accurate CPU prediction
- ✅ Proactive scaling (before demand hits)
- ✅ Real-time web dashboard
- ✅ AWS Lambda automation

---

## 🚀 5-Minute Local Setup

### Step 1: Clone & Install (2 min)
```bash
# Clone repository
git clone https://github.com/GannojiSathvik/Intelligent-Cloud-Scaling-System-Architecture.git
cd Intelligent-Cloud-Scaling-System-Architecture

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train Model (2 min)
```bash
# Train the optimized model
python train_model_optimized.py

# Output: lstm_model_optimized.pth (78.87% accuracy)
```

### Step 3: Make Predictions (1 min)
```bash
# Predict CPU for next 5 minutes
python predict.py

# Output:
# ==============================
#   PREDICTED CPU UTILIZATION: 56.72%
# ==============================
```

**🎉 Done! Your model is trained and making predictions!**

---

## ☁️ AWS Deployment (15 minutes)

### Prerequisites
- AWS Account
- AWS CLI configured (`aws configure`)

### Step 1: Create S3 Bucket
```bash
# Replace YOUR-NAME with your identifier
aws s3 mb s3://intelligent-scaling-demo-YOUR-NAME
```

### Step 2: Upload Files
```bash
# Upload model and data
aws s3 cp lstm_model_optimized.pth s3://intelligent-scaling-demo-YOUR-NAME/models/
aws s3 cp business_calendar.json s3://intelligent-scaling-demo-YOUR-NAME/
aws s3 cp multi_metric_data.csv s3://intelligent-scaling-demo-YOUR-NAME/
```

### Step 3: Deploy Lambda Functions
```bash
# Package DataCollector
cd lambda_functions
zip data_collector.zip DataCollectorFunction.py
aws lambda create-function \
  --function-name IntelligentScaling-DataCollector \
  --runtime python3.9 \
  --handler DataCollectorFunction.lambda_handler \
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
  --zip-file fileb://data_collector.zip \
  --timeout 30

# Package ScalingLogic
zip scaling_logic.zip ScalingLogicFunction.py
aws lambda create-function \
  --function-name IntelligentScaling-ScalingLogic \
  --runtime python3.9 \
  --handler ScalingLogicFunction.lambda_handler \
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
  --zip-file fileb://scaling_logic.zip \
  --timeout 30

# Package Dashboard API
zip dashboard_api.zip GetDashboardDataFunction.py
aws lambda create-function \
  --function-name IntelligentScaling-DashboardAPI \
  --runtime python3.9 \
  --handler GetDashboardDataFunction.lambda_handler \
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
  --zip-file fileb://dashboard_api.zip \
  --timeout 30
```

### Step 4: Schedule Lambda Functions
```bash
# Schedule DataCollector (every 5 minutes)
aws events put-rule \
  --name IntelligentScaling-DataCollection \
  --schedule-expression "rate(5 minutes)"

aws events put-targets \
  --rule IntelligentScaling-DataCollection \
  --targets "Id"="1","Arn"="arn:aws:lambda:REGION:ACCOUNT:function:IntelligentScaling-DataCollector"

# Schedule ScalingLogic (every 5 minutes)
aws events put-rule \
  --name IntelligentScaling-Scaling \
  --schedule-expression "rate(5 minutes)"

aws events put-targets \
  --rule IntelligentScaling-Scaling \
  --targets "Id"="1","Arn"="arn:aws:lambda:REGION:ACCOUNT:function:IntelligentScaling-ScalingLogic"
```

### Step 5: Deploy Dashboard
```bash
# Create API Gateway for Dashboard
aws apigateway create-rest-api --name IntelligentScalingDashboard

# Deploy index.html to S3
aws s3 cp index.html s3://your-dashboard-bucket/ --acl public-read
aws s3 website s3://your-dashboard-bucket/ --index-document index.html
```

**🎉 AWS Deployment Complete!**

---

## 📊 View Dashboard

### Local Testing
```bash
# Open index.html in browser (update API endpoint first)
open index.html  # macOS
# or
start index.html  # Windows
```

### Production
```
http://your-dashboard-bucket.s3-website-region.amazonaws.com
```

---

## 🧪 Test the System

### 1. Check Data Collection
```bash
aws s3 ls s3://intelligent-scaling-demo-YOUR-NAME/multi_metric_data.csv
```

### 2. Check Model Predictions
```bash
aws lambda invoke \
  --function-name IntelligentScaling-ScalingLogic \
  --payload '{}' \
  response.json
  
cat response.json
```

### 3. Verify Scaling
```bash
aws autoscaling describe-auto-scaling-groups \
  --auto-scaling-group-names intelligent-scaling-asg
```

---

## 📈 Monitor Performance

### CloudWatch Logs
```bash
# View DataCollector logs
aws logs tail /aws/lambda/IntelligentScaling-DataCollector --follow

# View ScalingLogic logs
aws logs tail /aws/lambda/IntelligentScaling-ScalingLogic --follow
```

### Dashboard Metrics
Open your dashboard URL to see:
- Current CPU utilization
- Server count
- Network traffic
- Request count
- Historical trends

---

## 🎓 Next Steps

### Improve Model Accuracy
```bash
# Train with advanced techniques
python train_model_advanced.py

# Evaluate all models
python evaluate_advanced.py
```

### Customize Business Logic
1. Edit `business_calendar.json` for your events
2. Adjust thresholds in `ScalingLogicFunction.py`:
   ```python
   SCALE_UP_THRESHOLD = 70    # Default: 70%
   SCALE_DOWN_THRESHOLD = 35  # Default: 35%
   ```

### Add Features
- Implement cost tracking
- Add email/SMS alerts
- Create mobile app
- Add multi-region support

---

## 🐛 Troubleshooting

### Model Not Found Error
```bash
# Ensure model is uploaded to S3
aws s3 cp lstm_model_optimized.pth s3://YOUR-BUCKET/models/
```

### Lambda Timeout
```bash
# Increase timeout to 60 seconds
aws lambda update-function-configuration \
  --function-name IntelligentScaling-ScalingLogic \
  --timeout 60
```

### Permission Errors
```bash
# Ensure Lambda has correct IAM permissions:
# - CloudWatch read
# - S3 read/write
# - Auto Scaling modify
# - Lambda invoke
```

### Dashboard Not Loading
1. Check API Gateway endpoint URL
2. Enable CORS in Lambda function
3. Verify S3 bucket is public (if using S3 hosting)

---

## 📚 Resources

### Documentation
- [README.md](README.md) - Full documentation
- [STRUCTURE.md](STRUCTURE.md) - File organization
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guide

### Training Guides
- [ACCURACY_IMPROVEMENT_GUIDE.md](docs/ACCURACY_IMPROVEMENT_GUIDE.md)
- [ADVANCED_TECHNIQUES_RESULTS.md](docs/ADVANCED_TECHNIQUES_RESULTS.md)

### AWS Guides
- [deployment_guide.md](docs/deployment_guide.md)

---

## 💡 Pro Tips

### Optimize Costs
```python
# Use smaller instances during off-peak
# Scale down aggressively when CPU < 20%
# Set min instances to 1 (not 2)
```

### Improve Accuracy
```python
# Retrain weekly with fresh data
# Add more features (user sessions, DB queries)
# Use ensemble of 3-5 models
```

### Scale for Production
```python
# Use SageMaker for model training
# Implement A/B testing for models
# Add model versioning
# Set up CloudWatch alarms
```

---

## 🎯 Success Checklist

- [ ] Model trained locally (78%+ accuracy)
- [ ] Predictions working
- [ ] S3 bucket created
- [ ] Lambda functions deployed
- [ ] Scheduled triggers configured
- [ ] Dashboard accessible
- [ ] Auto Scaling Group created
- [ ] System tested end-to-end

---

## 📞 Need Help?

- **Documentation**: Check [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/GannojiSathvik/Intelligent-Cloud-Scaling-System-Architecture/issues)
- **Questions**: [GitHub Discussions](https://github.com/GannojiSathvik/Intelligent-Cloud-Scaling-System-Architecture/discussions)

---

**🚀 You're all set! Your intelligent cloud scaling system is running!**

Made with ❤️ by Sathvik
