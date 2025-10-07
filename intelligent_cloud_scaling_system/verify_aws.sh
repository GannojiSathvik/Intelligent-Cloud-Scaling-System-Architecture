#!/bin/bash

# AWS Services Verification Script for Professor Demonstration
# Intelligent Cloud Scaling System

echo "================================================================================"
echo "  🔍 AWS INTELLIGENT CLOUD SCALING SYSTEM - VERIFICATION"
echo "  $(date)"
echo "================================================================================"

# Configuration
S3_BUCKET="my-intelligent-scaling-data-bucket"
ASG_NAME="intelligent-scaling-demo-sathvik"
REGION="us-east-1"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
        ((pass_count++))
    else
        echo -e "${RED}❌ $2${NC}"
        ((fail_count++))
    fi
}

echo ""
echo "================================================================================"
echo "  🔐 AWS ACCOUNT VERIFICATION"
echo "================================================================================"

aws sts get-caller-identity --output table
if [ $? -eq 0 ]; then
    print_status 0 "AWS credentials configured"
else
    print_status 1 "AWS credentials not configured"
    echo "Run: aws configure"
    exit 1
fi

echo ""
echo "================================================================================"
echo "  📦 S3 BUCKET STATUS"
echo "================================================================================"

# Check if bucket exists
aws s3 ls s3://$S3_BUCKET > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_status 0 "S3 Bucket exists: $S3_BUCKET"
    
    # List files
    echo ""
    echo "Files in bucket:"
    aws s3 ls s3://$S3_BUCKET --recursive --human-readable --summarize | tail -20
    
    # Check for data file
    aws s3 ls s3://$S3_BUCKET/multi_metric_data.csv > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_status 0 "Data file exists: multi_metric_data.csv"
        file_size=$(aws s3 ls s3://$S3_BUCKET/multi_metric_data.csv | awk '{print $3}')
        echo "   Size: $file_size bytes"
    else
        print_status 1 "Data file not found: multi_metric_data.csv"
    fi
    
    # Check for model file
    aws s3 ls s3://$S3_BUCKET/models/lstm_model.pth > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_status 0 "Model file exists: models/lstm_model.pth"
    else
        print_status 1 "Model file not found: models/lstm_model.pth"
    fi
else
    print_status 1 "S3 Bucket does not exist: $S3_BUCKET"
    echo ""
    echo "To create bucket, run:"
    echo "  aws s3 mb s3://$S3_BUCKET"
fi

echo ""
echo "================================================================================"
echo "  ⚡ LAMBDA FUNCTIONS STATUS"
echo "================================================================================"

# Check DataCollectorFunction
echo ""
echo "Checking DataCollectorFunction..."
aws lambda get-function --function-name DataCollectorFunction > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_status 0 "DataCollectorFunction deployed"
    aws lambda get-function --function-name DataCollectorFunction --query 'Configuration.[Runtime,MemorySize,Timeout,LastModified]' --output table
    
    # Check recent invocations
    echo "   Recent executions:"
    aws logs tail /aws/lambda/DataCollectorFunction --since 1h --format short 2>/dev/null | head -5
else
    print_status 1 "DataCollectorFunction not deployed"
fi

# Check ScalingLogicFunction
echo ""
echo "Checking ScalingLogicFunction..."
aws lambda get-function --function-name ScalingLogicFunction > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_status 0 "ScalingLogicFunction deployed"
    aws lambda get-function --function-name ScalingLogicFunction --query 'Configuration.[Runtime,MemorySize,Timeout,LastModified]' --output table
    
    # Check recent invocations
    echo "   Recent executions:"
    aws logs tail /aws/lambda/ScalingLogicFunction --since 1h --format short 2>/dev/null | head -5
else
    print_status 1 "ScalingLogicFunction not deployed"
fi

echo ""
echo "================================================================================"
echo "  🔄 AUTO SCALING GROUP STATUS"
echo "================================================================================"

aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names $ASG_NAME > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_status 0 "Auto Scaling Group exists: $ASG_NAME"
    echo ""
    aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names $ASG_NAME \
        --query 'AutoScalingGroups[0].[DesiredCapacity,MinSize,MaxSize]' --output table
    
    echo ""
    echo "Instances:"
    aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names $ASG_NAME \
        --query 'AutoScalingGroups[0].Instances[*].[InstanceId,HealthStatus,LifecycleState]' --output table
    
    echo ""
    echo "Recent Scaling Activities:"
    aws autoscaling describe-scaling-activities --auto-scaling-group-name $ASG_NAME \
        --max-records 5 --query 'Activities[*].[StartTime,Description,StatusCode]' --output table
else
    print_status 1 "Auto Scaling Group not found: $ASG_NAME"
fi

echo ""
echo "================================================================================"
echo "  ⏰ EVENTBRIDGE RULES STATUS"
echo "================================================================================"

# Check for EventBridge rules
rules=$(aws events list-rules --name-prefix "ScalingLogic" --query 'Rules[*].Name' --output text)
if [ -n "$rules" ]; then
    print_status 0 "EventBridge rules found"
    for rule in $rules; do
        echo ""
        echo "Rule: $rule"
        aws events describe-rule --name $rule --query '[ScheduleExpression,State]' --output table
        
        # Check targets
        target_count=$(aws events list-targets-by-rule --rule $rule --query 'length(Targets)' --output text)
        echo "   Targets: $target_count"
    done
else
    print_status 1 "No EventBridge rules found"
fi

echo ""
echo "================================================================================"
echo "  📊 CLOUDWATCH METRICS STATUS"
echo "================================================================================"

# Check for ASG metrics
metrics=$(aws cloudwatch list-metrics --namespace "AWS/EC2" \
    --dimensions Name=AutoScalingGroupName,Value=$ASG_NAME \
    --query 'Metrics[*].MetricName' --output text | head -5)

if [ -n "$metrics" ]; then
    print_status 0 "CloudWatch metrics available"
    echo "   Metrics: $metrics"
else
    print_status 1 "No CloudWatch metrics found"
fi

echo ""
echo "================================================================================"
echo "  🎯 SUMMARY"
echo "================================================================================"

total=$((pass_count + fail_count))
echo ""
echo "System Status: $pass_count/$total components operational"
echo ""

if [ $pass_count -eq $total ]; then
    echo -e "${GREEN}🎉 ALL SYSTEMS OPERATIONAL - READY FOR DEMONSTRATION!${NC}"
elif [ $pass_count -ge $((total * 7 / 10)) ]; then
    echo -e "${YELLOW}⚠️  PARTIAL DEPLOYMENT - Some components need attention${NC}"
else
    echo -e "${RED}❌ SYSTEM NOT READY - Multiple components need deployment${NC}"
fi

echo ""
echo "================================================================================"
echo "  📋 DEMONSTRATION CHECKLIST"
echo "================================================================================"
echo ""
echo "For Professor Demonstration, ensure:"
echo "  [ ] S3 bucket exists with data collection"
echo "  [ ] Lambda functions deployed and executing"
echo "  [ ] Auto Scaling Group running with instances"
echo "  [ ] EventBridge triggers configured"
echo "  [ ] CloudWatch logs showing activity"
echo "  [ ] Model trained and uploaded to S3"
echo ""
echo "Report saved to: aws_verification_report.txt"
echo "================================================================================"

# Save report
{
    echo "AWS Verification Report"
    echo "Generated: $(date)"
    echo "Passed: $pass_count/$total"
    echo ""
    echo "Account: $(aws sts get-caller-identity --query Account --output text)"
    echo "Region: $(aws configure get region)"
} > aws_verification_report.txt

exit 0
