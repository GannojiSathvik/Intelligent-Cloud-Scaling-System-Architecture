import boto3
import json
import pandas as pd
import io

# --- Configuration ---
# ======================== These values are from your script =========================
# Use the same S3 bucket and Region as your other scripts.
S3_BUCKET_NAME = "my-intelligent-scaling-data-bucket"
AWS_REGION = "us-east-2" 
# FINAL UPDATE: Using the exact name from your Auto Scaling Group configuration.
ASG_NAME = 'IntelligentScaling-ASG' 
# ======================================================================

DATA_FILE_KEY = "multi_metric_data.csv"
CALENDAR_FILE_KEY = "config/business_calendar.json"

s3_client = boto3.client('s3', region_name=AWS_REGION)
autoscaling_client = boto3.client('autoscaling', region_name=AWS_REGION)

def lambda_handler(event, context):
    """
    This function is the backend for the GUI dashboard.
    It fetches live data and returns it as a JSON object.
    """
    try:
        # --- 1. Get Historical Data ---
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=DATA_FILE_KEY)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))

        # This code now correctly reads 'cpu_utilization' and renames it to 'cpu' for the dashboard.
        if 'cpu_utilization' in df.columns:
            historical_data = df.tail(100)[['timestamp', 'cpu_utilization']].rename(columns={'cpu_utilization': 'cpu'}).to_dict('records')
        elif 'cpu' in df.columns:
            historical_data = df.tail(100)[['timestamp', 'cpu']].to_dict('records')
        else:
            raise KeyError("Neither 'cpu' nor 'cpu_utilization' column found in the CSV file.")


        # --- 2. Get Business Context ---
        try:
            cal_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=CALENDAR_FILE_KEY)
            calendar = json.loads(cal_obj['Body'].read().decode('utf-8'))
            is_sale_active = calendar.get('is_sale_active', 0)
        except s3_client.exceptions.NoSuchKey:
            is_sale_active = 0 # Default to inactive if calendar doesn't exist

        # --- 3. Get Current Server Count ---
        response = autoscaling_client.describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])
        if not response['AutoScalingGroups']:
            raise ValueError(f"Auto Scaling Group '{ASG_NAME}' not found. Please check the name and region.")
        current_server_count = response['AutoScalingGroups'][0]['DesiredCapacity']

        # --- 4. Format the final JSON response ---
        dashboard_data = {
            "current_server_count": current_server_count,
            "is_sale_active": is_sale_active,
            "historical_data": historical_data
        }

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET'
            },
            'body': json.dumps(dashboard_data)
        }

    except KeyError as e:
        print(f"Error: A column was not found. Please check your CSV headers. Details: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f"Column not found in CSV: {e}"})
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

