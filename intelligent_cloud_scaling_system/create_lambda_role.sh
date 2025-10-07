#!/bin/bash

# Create IAM Role for Lambda Functions

ROLE_NAME="IntelligentScalingLambdaRole"

echo "Creating IAM Role for Lambda Functions..."

# Create trust policy
cat > trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
aws iam create-role \
  --role-name $ROLE_NAME \
  --assume-role-policy-document file://trust-policy.json \
  --description "Execution role for Intelligent Scaling Lambda functions"

# Attach AWS managed policies
aws iam attach-role-policy \
  --role-name $ROLE_NAME \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Create custom policy for S3 and Auto Scaling
cat > lambda-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-intelligent-scaling-data-bucket",
        "arn:aws:s3:::my-intelligent-scaling-data-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "autoscaling:DescribeAutoScalingGroups",
        "autoscaling:SetDesiredCapacity",
        "autoscaling:DescribeScalingActivities"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:ListMetrics"
      ],
      "Resource": "*"
    }
  ]
}
EOF

# Create and attach custom policy
aws iam put-role-policy \
  --role-name $ROLE_NAME \
  --policy-name IntelligentScalingPolicy \
  --policy-document file://lambda-policy.json

# Get role ARN
ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)

echo ""
echo "✅ Role created successfully!"
echo "Role ARN: $ROLE_ARN"
echo ""
echo "Update setup_aws_complete.sh with this ARN:"
echo "LAMBDA_ROLE_ARN=\"$ROLE_ARN\""

# Cleanup
rm -f trust-policy.json lambda-policy.json

# Wait for role to propagate
echo ""
echo "Waiting 10 seconds for role to propagate..."
sleep 10
echo "✅ Ready to deploy Lambda functions"
