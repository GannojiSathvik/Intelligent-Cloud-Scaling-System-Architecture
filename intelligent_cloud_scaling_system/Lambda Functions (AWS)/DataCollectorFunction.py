import json
import boto3
import os
from datetime import datetime, timedelta
import csv
import io

# --- Configuration ---
S3_BUCKET_NAME = "sathvik-scaling-data-us-east-2"
DATA_FILE_KEY = "multi_metric_data.csv"
CALENDAR_FILE_KEY = "config/business_calendar.json" # Corrected path based on GetDashboardDataFunction.py
AUTO_SCALING_GROUP_NAME = "IntelligentScaling-ASG"
CLOUDWATCH_METRIC_NAMESPACE = "AWS/EC2"

s3_client = boto3.client('s3')
cloudwatch_client = boto3.client('cloudwatch')

def lambda_handler(event, context):
    """
    This Lambda function collects metrics from CloudWatch and business context from S3,
    then appends the new data to a CSV file in S3.
    """
    try:
        # --- 1. Get Business Context from S3 ---
        is_sale_active = 0 # Default value
        try:
            calendar_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=CALENDAR_FILE_KEY)
            calendar_content = json.loads(calendar_obj['Body'].read().decode('utf-8'))
            is_sale_active = calendar_content.get("is_sale_active", 0)
            print(f"Successfully read business calendar. is_sale_active = {is_sale_active}")
        except s3_client.exceptions.NoSuchKey:
            print(f"'{CALENDAR_FILE_KEY}' not found in S3. Using default value for is_sale_active.")
        except Exception as e:
            print(f"Error reading '{CALENDAR_FILE_KEY}' from S3: {e}")
            # Continue with default value

        # --- 2. Get Metrics from CloudWatch ---
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)

        def get_metric_average(metric_name):
            response = cloudwatch_client.get_metric_statistics(
                Namespace=CLOUDWATCH_METRIC_NAMESPACE,
                MetricName=metric_name,
                Dimensions=[{'Name': 'AutoScalingGroupName', 'Value': AUTO_SCALING_GROUP_NAME}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )
            if response['Datapoints']:
                return response['Datapoints'][0]['Average']
            return 0 # Return 0 if no data is available

        cpu_utilization = get_metric_average('CPUUtilization')
        network_in = get_metric_average('NetworkIn')
        # request_count is simulated here as it's not a standard CloudWatch metric for ASGs
        # In a real system, this would come from a load balancer or application metric.
        request_count = 0 # Placeholder

        print(f"Collected metrics: CPU={cpu_utilization:.2f}%, NetworkIn={network_in:.2f} bytes, IsSaleActive={is_sale_active}")

        # --- 3. Append Data to CSV in S3 ---
        new_row = [
            datetime.now().isoformat(),
            cpu_utilization,
            network_in,
            request_count,
            is_sale_active
        ]
        print(f"New row to append: {new_row}")

        try:
            # Get the existing CSV file
            csv_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=DATA_FILE_KEY)
            old_content = csv_obj['Body'].read().decode('utf-8')
            print(f"Existing CSV content length: {len(old_content)} bytes")
        except s3_client.exceptions.NoSuchKey:
            print(f"'{DATA_FILE_KEY}' not found. Creating a new file with headers.")
            old_content = "timestamp,cpu_utilization,network_in,request_count,is_sale_active\n"
        except Exception as e:
            print(f"Error getting existing CSV from S3: {e}")
            # Fallback to creating new file if there's an error reading existing one
            old_content = "timestamp,cpu_utilization,network_in,request_count,is_sale_active\n"


        # Append the new row
        output = io.StringIO()
        output.write(old_content)
        writer = csv.writer(output)
        writer.writerow(new_row)
        new_content = output.getvalue()
        print(f"New CSV content length: {len(new_content)} bytes")

        # Upload the updated CSV back to S3
        print(f"Attempting to upload updated CSV to s3://{S3_BUCKET_NAME}/{DATA_FILE_KEY}")
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=DATA_FILE_KEY, Body=new_content)
        print(f"Successfully uploaded updated CSV to s3://{S3_BUCKET_NAME}/{DATA_FILE_KEY}")

        return {
            'statusCode': 200,
            'body': json.dumps('Data collection successful!')
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error during data collection: {str(e)}')
        }
