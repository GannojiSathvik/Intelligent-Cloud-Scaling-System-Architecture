#!/bin/bash

# Complete AWS Setup Script for Intelligent Cloud Scaling System
# This script sets up all AWS resources needed for the professor demonstration

set -e  # Exit on error

# Configuration
S3_BUCKET="my-intelligent-scaling-data-bucket"
ASG_NAME="intelligent-scaling-demo-sathvik"
REGION="us-east-1"
LAMBDA_ROLE_ARN="arn:aws:iam::780139019966:role/IntelligentScalingLambdaRole"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "================================================================================"
echo "  🚀 AWS INTELLIGENT CLOUD SCALING SYSTEM - COMPLETE SETUP"
echo "  $(date)"
echo "================================================================================"

# Step 1: Create S3 Bucket
echo ""
echo -e "${BLUE}📦 Step 1: Creating S3 Bucket...${NC}"
if aws s3 ls s3://$S3_BUCKET 2>/dev/null; then
    echo -e "${GREEN}✅ Bucket already exists: $S3_BUCKET${NC}"
else
    aws s3 mb s3://$S3_BUCKET
    echo -e "${GREEN}✅ Created bucket: $S3_BUCKET${NC}"
fi

# Step 2: Upload training data
echo ""
echo -e "${BLUE}📤 Step 2: Uploading training data...${NC}"
if [ -f "multi_metric_data.csv" ]; then
    aws s3 cp multi_metric_data.csv s3://$S3_BUCKET/multi_metric_data.csv
    echo -e "${GREEN}✅ Uploaded multi_metric_data.csv${NC}"
else
    echo -e "${YELLOW}⚠️  multi_metric_data.csv not found, skipping${NC}"
fi

# Step 3: Upload trained model
echo ""
echo -e "${BLUE}🤖 Step 3: Uploading trained model...${NC}"
if [ -f "lstm_model_advanced.pth" ]; then
    aws s3 cp lstm_model_advanced.pth s3://$S3_BUCKET/models/lstm_model.pth
    echo -e "${GREEN}✅ Uploaded lstm_model_advanced.pth as lstm_model.pth${NC}"
elif [ -f "lstm_model.pth" ]; then
    aws s3 cp lstm_model.pth s3://$S3_BUCKET/models/lstm_model.pth
    echo -e "${GREEN}✅ Uploaded lstm_model.pth${NC}"
else
    echo -e "${YELLOW}⚠️  No model file found, skipping${NC}"
fi

# Step 4: Upload business calendar
echo ""
echo -e "${BLUE}📅 Step 4: Uploading business calendar...${NC}"
if [ -f "business_calendar.json" ]; then
    aws s3 cp business_calendar.json s3://$S3_BUCKET/business_calendar.json
    echo -e "${GREEN}✅ Uploaded business_calendar.json${NC}"
else
    echo '{"is_sale_active": 0}' > business_calendar.json
    aws s3 cp business_calendar.json s3://$S3_BUCKET/business_calendar.json
    echo -e "${GREEN}✅ Created and uploaded business_calendar.json${NC}"
fi

# Step 5: Package Lambda functions
echo ""
echo -e "${BLUE}📦 Step 5: Packaging Lambda functions...${NC}"

# Package DataCollectorFunction
if [ -f "DataCollectorFunction.py" ]; then
    zip -q DataCollectorFunction.zip DataCollectorFunction.py
    echo -e "${GREEN}✅ Packaged DataCollectorFunction${NC}"
fi

# Package ScalingLogicFunction
if [ -f "ScalingLogicFunction.py" ]; then
    zip -q ScalingLogicFunction.zip ScalingLogicFunction.py
    echo -e "${GREEN}✅ Packaged ScalingLogicFunction${NC}"
fi

# Step 6: Deploy DataCollectorFunction
echo ""
echo -e "${BLUE}⚡ Step 6: Deploying DataCollectorFunction...${NC}"
if aws lambda get-function --function-name DataCollectorFunction 2>/dev/null; then
    aws lambda update-function-code \
        --function-name DataCollectorFunction \
        --zip-file fileb://DataCollectorFunction.zip > /dev/null
    echo -e "${GREEN}✅ Updated DataCollectorFunction${NC}"
else
    aws lambda create-function \
        --function-name DataCollectorFunction \
        --runtime python3.9 \
        --role $LAMBDA_ROLE_ARN \
        --handler DataCollectorFunction.lambda_handler \
        --zip-file fileb://DataCollectorFunction.zip \
        --timeout 30 \
        --memory-size 256 \
        --description "Collects metrics from CloudWatch and business context" > /dev/null
    echo -e "${GREEN}✅ Created DataCollectorFunction${NC}"
fi

# Step 7: Deploy ScalingLogicFunction
echo ""
echo -e "${BLUE}⚡ Step 7: Deploying ScalingLogicFunction...${NC}"
if aws lambda get-function --function-name ScalingLogicFunction 2>/dev/null; then
    aws lambda update-function-code \
        --function-name ScalingLogicFunction \
        --zip-file fileb://ScalingLogicFunction.zip > /dev/null
    echo -e "${GREEN}✅ Updated ScalingLogicFunction${NC}"
else
    aws lambda create-function \
        --function-name ScalingLogicFunction \
        --runtime python3.9 \
        --role $LAMBDA_ROLE_ARN \
        --handler ScalingLogicFunction.lambda_handler \
        --zip-file fileb://ScalingLogicFunction.zip \
        --timeout 60 \
        --memory-size 512 \
        --description "Predictive scaling logic using LSTM model" > /dev/null
    echo -e "${GREEN}✅ Created ScalingLogicFunction${NC}"
fi

# Note: PyTorch layer needs to be added manually via console or with specific ARN

# Step 8: Create EventBridge rules
echo ""
echo -e "${BLUE}⏰ Step 8: Creating EventBridge triggers...${NC}"

# DataCollector trigger (every 5 minutes)
RULE_NAME="DataCollectorSchedule"
if aws events describe-rule --name $RULE_NAME 2>/dev/null; then
    echo -e "${GREEN}✅ EventBridge rule already exists: $RULE_NAME${NC}"
else
    aws events put-rule \
        --name $RULE_NAME \
        --schedule-expression "rate(5 minutes)" \
        --state ENABLED \
        --description "Trigger data collection every 5 minutes" > /dev/null
    
    LAMBDA_ARN=$(aws lambda get-function --function-name DataCollectorFunction --query 'Configuration.FunctionArn' --output text)
    
    aws events put-targets \
        --rule $RULE_NAME \
        --targets "Id"="1","Arn"="$LAMBDA_ARN" > /dev/null
    
    aws lambda add-permission \
        --function-name DataCollectorFunction \
        --statement-id ${RULE_NAME}-Permission \
        --action 'lambda:InvokeFunction' \
        --principal events.amazonaws.com \
        --source-arn $(aws events describe-rule --name $RULE_NAME --query 'Arn' --output text) 2>/dev/null || true
    
    echo -e "${GREEN}✅ Created EventBridge rule: $RULE_NAME${NC}"
fi

# ScalingLogic trigger (every 5 minutes)
RULE_NAME="ScalingLogicSchedule"
if aws events describe-rule --name $RULE_NAME 2>/dev/null; then
    echo -e "${GREEN}✅ EventBridge rule already exists: $RULE_NAME${NC}"
else
    aws events put-rule \
        --name $RULE_NAME \
        --schedule-expression "rate(5 minutes)" \
        --state ENABLED \
        --description "Trigger scaling logic every 5 minutes" > /dev/null
    
    LAMBDA_ARN=$(aws lambda get-function --function-name ScalingLogicFunction --query 'Configuration.FunctionArn' --output text)
    
    aws events put-targets \
        --rule $RULE_NAME \
        --targets "Id"="1","Arn"="$LAMBDA_ARN" > /dev/null
    
    aws lambda add-permission \
        --function-name ScalingLogicFunction \
        --statement-id ${RULE_NAME}-Permission \
        --action 'lambda:InvokeFunction' \
        --principal events.amazonaws.com \
        --source-arn $(aws events describe-rule --name $RULE_NAME --query 'Arn' --output text) 2>/dev/null || true
    
    echo -e "${GREEN}✅ Created EventBridge rule: $RULE_NAME${NC}"
fi

# Cleanup
rm -f DataCollectorFunction.zip ScalingLogicFunction.zip

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ SETUP COMPLETE!${NC}"
echo "================================================================================"
echo ""
echo "📊 Deployed Resources:"
echo "  • S3 Bucket: $S3_BUCKET"
echo "  • Lambda Functions: DataCollectorFunction, ScalingLogicFunction"
echo "  • EventBridge Rules: Running every 5 minutes"
echo ""
echo "⚠️  IMPORTANT: Add PyTorch Layer to ScalingLogicFunction"
echo "  1. Go to AWS Lambda Console"
echo "  2. Open ScalingLogicFunction"
echo "  3. Add Layer with ARN: arn:aws:lambda:us-east-1:770693421928:layer:Klayers-p39-torch:1"
echo ""
echo "🔗 Quick Links:"
echo "  • S3: https://s3.console.aws.amazon.com/s3/buckets/$S3_BUCKET"
echo "  • Lambda: https://console.aws.amazon.com/lambda/home#/functions"
echo "  • EventBridge: https://console.aws.amazon.com/events/home#/rules"
echo ""
echo "📝 Next: Run './verify_aws.sh' to check system status"
echo "================================================================================"
