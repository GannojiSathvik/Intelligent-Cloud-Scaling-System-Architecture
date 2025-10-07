# 🚀 Quick Demo Commands - Professor Presentation

## 📊 Show Model Performance (30 seconds)

```bash
# Display evaluation results
cat evaluation_results.json | python -m json.tool

# Key metrics to highlight:
# - Best Model: 78.51% accuracy
# - R² Score: 0.9328
# - MAE: 4.34%
```

## ☁️ Verify AWS Deployment (1 minute)

```bash
# Run complete verification
./verify_aws.sh

# Expected: 8/9 components operational
```

## 📦 Show S3 Bucket Contents (30 seconds)

```bash
# List all files
aws s3 ls s3://my-intelligent-scaling-data-bucket/ --recursive --human-readable

# Preview data file (last 10 rows)
aws s3 cp s3://my-intelligent-scaling-data-bucket/multi_metric_data.csv - | tail -10

# Check model file
aws s3 ls s3://my-intelligent-scaling-data-bucket/models/
```

## ⚡ Show Lambda Functions (30 seconds)

```bash
# List functions
aws lambda list-functions --query 'Functions[?contains(FunctionName, `Scaling`) || contains(FunctionName, `Collector`)].{Name:FunctionName,Runtime:Runtime,Memory:MemorySize}' --output table

# Get function details
aws lambda get-function --function-name ScalingLogicFunction --query 'Configuration.[FunctionName,Runtime,MemorySize,Timeout,LastModified]' --output table
```

## ⏰ Show EventBridge Triggers (30 seconds)

```bash
# List scheduled rules
aws events list-rules --query 'Rules[?contains(Name, `Schedule`)].{Name:Name,Schedule:ScheduleExpression,State:State}' --output table
```

## 📈 Monitor Live Activity (if time permits)

```bash
# Watch DataCollector logs
aws logs tail /aws/lambda/DataCollectorFunction --follow --since 10m

# Watch ScalingLogic logs
aws logs tail /aws/lambda/ScalingLogicFunction --follow --since 10m
```

## 🔍 Quick System Check

```bash
# One-liner status check
echo "S3: $(aws s3 ls s3://my-intelligent-scaling-data-bucket/ 2>&1 | grep -q 'multi_metric_data.csv' && echo '✅' || echo '❌')" && \
echo "Lambda DataCollector: $(aws lambda get-function --function-name DataCollectorFunction 2>&1 | grep -q 'FunctionName' && echo '✅' || echo '❌')" && \
echo "Lambda ScalingLogic: $(aws lambda get-function --function-name ScalingLogicFunction 2>&1 | grep -q 'FunctionName' && echo '✅' || echo '❌')" && \
echo "EventBridge: $(aws events list-rules --name-prefix 'Scaling' 2>&1 | grep -q 'Rules' && echo '✅' || echo '❌')"
```

## 🎯 Key Talking Points

### When showing ML model:
- "Achieved 78.51% accuracy using LSTM with Attention mechanism"
- "Trained on 4,293 data points with 16 engineered features"
- "Ensemble of 5 models for robust predictions"

### When showing AWS infrastructure:
- "Fully serverless architecture using Lambda and S3"
- "Automated data collection and prediction every 5 minutes"
- "Proactive scaling before demand spikes occur"

### When showing system architecture:
- "CloudWatch metrics → Lambda → ML Model → Auto Scaling"
- "Context-aware: incorporates business events and temporal patterns"
- "Production-ready: runs autonomously 24/7"

## 📱 AWS Console Quick Links

```bash
# Open in browser:
# S3: https://s3.console.aws.amazon.com/s3/buckets/my-intelligent-scaling-data-bucket
# Lambda: https://console.aws.amazon.com/lambda/home?region=ap-south-1#/functions
# EventBridge: https://console.aws.amazon.com/events/home?region=ap-south-1#/rules
```

## 🎬 15-Minute Demo Flow

### Minutes 0-5: ML Model
```bash
python evaluate_advanced.py  # Show full evaluation
cat evaluation_results.json | python -m json.tool  # Show metrics
```

### Minutes 5-10: AWS Infrastructure
```bash
./verify_aws.sh  # Complete system check
aws s3 ls s3://my-intelligent-scaling-data-bucket/ --recursive --human-readable
aws lambda list-functions --output table
```

### Minutes 10-13: Architecture & Design
- Open PROFESSOR_DEMONSTRATION.md
- Show architecture diagram
- Explain data flow

### Minutes 13-15: Q&A
- Be ready to show CloudWatch logs
- Explain scaling thresholds
- Discuss cost optimization

## 🆘 Troubleshooting

### If Lambda shows no recent executions:
```bash
# Manually invoke to test
aws lambda invoke --function-name DataCollectorFunction output.json
cat output.json
```

### If S3 bucket is empty:
```bash
# Re-run setup
./setup_aws_complete.sh
```

### If verification fails:
```bash
# Check AWS credentials
aws sts get-caller-identity

# Check region
aws configure get region
```

## ✅ Pre-Demo Checklist

- [ ] Run `./verify_aws.sh` - Should show 8/9 operational
- [ ] Check `evaluation_results.json` exists
- [ ] Verify S3 bucket has 3 files (data, model, calendar)
- [ ] Confirm Lambda functions deployed
- [ ] Have `PROFESSOR_DEMONSTRATION.md` open
- [ ] Terminal ready with commands
- [ ] AWS Console tabs open (S3, Lambda, EventBridge)

## 🎉 Success Metrics to Highlight

| Metric | Value | Impact |
|--------|-------|--------|
| Model Accuracy | 78.51% | High prediction reliability |
| R² Score | 0.9328 | Explains 93% of variance |
| Components Deployed | 8/9 | 89% system completion |
| Automation | 100% | Fully autonomous operation |
| Cost | <$5/month | Highly cost-effective |

---

**Last Updated:** October 6, 2025, 11:13 PM IST  
**Status:** ✅ READY FOR DEMONSTRATION
