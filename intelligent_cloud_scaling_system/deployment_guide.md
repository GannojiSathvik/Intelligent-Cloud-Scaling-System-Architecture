# Deployment Guide: Intelligent Cloud Scaling System

This guide provides the steps to connect and deploy the Intelligent Cloud Scaling System to your AWS account.

### **Prerequisite: AWS CLI Configuration**

Ensure you have the AWS CLI installed and configured on your local machine. The `boto3` library used in these scripts will automatically use your configured credentials. You can configure the CLI by running:
```bash
aws configure
```
You will be prompted for your AWS Access Key ID, Secret Access Key, default region, and output format.

---

### **Step 1: Create an S3 Bucket**

This bucket will store your dataset, the business context file, and the trained machine learning model.

1.  Choose a unique name for your S3 bucket (e.g., `my-intelligent-scaling-data-bucket`).
2.  Create the bucket in your desired AWS region.

---

### **Step 2: Create an EC2 Auto Scaling Group (ASG)**

If you don't have one already, create a standard EC2 Auto Scaling Group.

1.  Create a Launch Template with your desired EC2 instance type and AMI.
2.  Create an Auto Scaling Group using this launch template.
3.  Set the initial **Min Size**, **Max Size**, and **Desired Capacity**.
4.  **Crucially, note down the exact name of your Auto Scaling Group.**

---

### **Step 3: Update Configuration in the Python Scripts**

You must replace the placeholder values in the scripts with your actual AWS resource names.

1.  **`train_model.py`**:
    *   Update `S3_BUCKET_NAME` to your bucket name.

2.  **`DataCollectorFunction.py`**:
    *   Update `S3_BUCKET_NAME` to your bucket name.
    *   Update `AUTO_SCALING_GROUP_NAME` to your ASG name.

3.  **`ScalingLogicFunction.py`**:
    *   Update `S3_BUCKET_NAME` to your bucket name.
    *   Update `AUTO_SCALING_GROUP_NAME` to your ASG name.

---

### **Step 4: Upload Initial Files to S3**

Upload the generated data and the business calendar to your S3 bucket.

1.  Navigate to the `intelligent_cloud_scaling_system` directory in your terminal.
2.  Run the following AWS CLI commands:
    ```bash
    # Replace 'your-s3-bucket-name' with your actual bucket name
    aws s3 cp multi_metric_data.csv s3://your-s3-bucket-name/multi_metric_data.csv
    aws s3 cp business_calendar.json s3://your-s3-bucket-name/business_calendar.json
    ```

---

### **Step 5: Train the Model and Upload to S3**

1.  In `train_model.py`, uncomment the lines in "Section 6" to enable uploading the model to S3.
2.  Run the script locally to train the model and upload `lstm_model.h5` to your S3 bucket:
    ```bash
    python3 train_model.py
    ```

---

### **Step 6: Create and Deploy the Lambda Functions**

You will create two Lambda functions. For each function, you need to create an IAM Role with the necessary permissions.

#### **A. IAM Role for `DataCollectorFunction`**

1.  In the IAM console, create a new Role.
2.  Select "AWS service" and "Lambda" as the use case.
3.  Create a new policy with the following JSON permissions. Replace `your-s3-bucket-name` with your bucket name.
    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject"
                ],
                "Resource": "arn:aws:s3:::your-s3-bucket-name/*"
            },
            {
                "Effect": "Allow",
                "Action": "cloudwatch:GetMetricStatistics",
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "arn:aws:logs:*:*:*"
            }
        ]
    }
    ```
4.  Attach this policy to your new role. Name the role something descriptive, like `DataCollectorLambdaRole`.

#### **B. IAM Role for `ScalingLogicFunction`**

1.  Create another IAM Role for Lambda.
2.  Create a new policy with the following JSON permissions. Replace placeholders accordingly.
    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::your-s3-bucket-name/*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "autoscaling:DescribeAutoScalingGroups",
                    "autoscaling:SetDesiredCapacity"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "arn:aws:logs:*:*:*"
            }
        ]
    }
    ```
3.  Attach this policy to the role. Name it something like `ScalingLogicLambdaRole`.

#### **C. Create the Lambda Functions**

For both `DataCollectorFunction.py` and `ScalingLogicFunction.py`:

1.  Go to the AWS Lambda console and create a new function.
2.  Choose "Author from scratch".
3.  Use a **Python 3.9** (or newer) runtime.
4.  Under "Permissions", choose "Use an existing role" and select the corresponding IAM role you just created.
5.  Copy the code from the `.py` file and paste it into the Lambda code editor.
6.  **For `ScalingLogicFunction`**, you will need to package the necessary libraries (TensorFlow, Pandas, scikit-learn) into a Lambda Layer, as these are not included in the default Lambda environment. This is an advanced step that involves creating a deployment package.
7.  **Set up Triggers**:
    *   For both functions, add a trigger.
    *   Select **EventBridge (CloudWatch Events)**.
    *   Create a new rule.
    *   Set the schedule expression to `rate(5 minutes)`.

---

After completing these steps, your intelligent scaling system will be fully operational on AWS.
