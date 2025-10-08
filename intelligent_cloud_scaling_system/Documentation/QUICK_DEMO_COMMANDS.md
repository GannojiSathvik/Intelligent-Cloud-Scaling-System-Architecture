# 🚀 Quick Demo Commands - Professor Presentation

## 📊 Show Model Performance (30 seconds)

```bash
# Display evaluation results
cat "Data Files/evaluation_results.json" | python -m json.tool

# Key metrics to highlight:
# - Best Model: 78.51% accuracy
# - R² Score: 0.9328
# - MAE: 4.34%

# Run complete verification
./verify_aws.sh

# Expected: 8/9 components operational


# List all files
aws s3 ls s3://my-intelligent-scaling-data-bucket/ --recursive --human-readable

# Preview data file (last 10 rows)
aws s3 cp s3://my-intelligent-scaling-data-bucket/multi_metric_data.csv - | tail -10

# Check model file
aws s3 ls s3://my-intelligent-scaling-data-bucket/models/

# List functions
aws lambda list-functions --query 'Functions[?contains(FunctionName, `Scaling`) || contains(FunctionName, `Collector`)].{Name:FunctionName,Runtime:Runtime,Memory:MemorySize}' --output table

# Get function details
aws lambda get-function --function-name ScalingLogicFunction --query 'Configuration.[FunctionName,Runtime,MemorySize,Timeout,LastModified]' --output table

# List scheduled rules
aws events list-rules --query 'Rules[?contains(Name, `Schedule`)].{Name:Name,Schedule:ScheduleExpression,State:State}' --output table


# Watch DataCollector logs
aws logs tail /aws/lambda/DataCollectorFunction --follow --since 10m

# Watch ScalingLogic logs
aws logs tail /aws/lambda/ScalingLogicFunction --follow --since 10m


# One-liner status check
echo "S3: $(aws s3 ls s3://my-intelligent-scaling-data-bucket/ 2>&1 | grep -q 'multi_metric_data.csv' && echo '✅' || echo '❌')" && \
echo "Lambda DataCollector: $(aws lambda get-function --function-name DataCollectorFunction 2>&1 | grep -q 'FunctionName' && echo '✅' || echo '❌')" && \
echo "Lambda ScalingLogic: $(aws lambda get-function --function-name ScalingLogicFunction 2>&1 | grep -q 'FunctionName' && echo '✅' || echo '❌')" && \
echo "EventBridge: $(aws events list-rules --name-prefix 'Scaling' 2>&1 | grep -q 'Rules' && echo '✅' || echo '❌')"