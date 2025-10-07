import json
import boto3
import pandas as pd
from io import StringIO
from datetime import datetime

# Configuration
S3_BUCKET_NAME = 'intelligent-scaling-demo-sathvik'
DATA_FILE_KEY = 'multi_metric_data.csv'
BUSINESS_CALENDAR_KEY = 'business_calendar.json'
AUTO_SCALING_GROUP_NAME = 'intelligent-scaling-asg'

s3_client = boto3.client('s3')
autoscaling_client = boto3.client('autoscaling')

def lambda_handler(event, context):
    """
    Lambda function to provide dashboard data for the Intelligent Cloud Scaling System.
    Returns current metrics, business context, and historical data.
    """
    try:
        # 1. Get current server count from Auto Scaling Group
        current_server_count = get_current_server_count()
        
        # 2. Get business context (is_sale_active)
        is_sale_active = get_business_context()
        
        # 3. Get historical metrics data (last 100 rows)
        historical_data = get_historical_metrics()
        
        # 4. Format and return response
        response_data = {
            "current_server_count": current_server_count,
            "is_sale_active": is_sale_active,
            "historical_data": historical_data,
            "last_updated": datetime.utcnow().isoformat() + "Z"
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',  # Enable CORS
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET, OPTIONS'
            },
            'body': json.dumps(response_data)
        }
        
    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'message': 'Failed to retrieve dashboard data'
            })
        }

def get_current_server_count():
    """
    Retrieves the current desired capacity (number of active servers) 
    from the Auto Scaling Group.
    """
    try:
        response = autoscaling_client.describe_auto_scaling_groups(
            AutoScalingGroupNames=[AUTO_SCALING_GROUP_NAME]
        )
        
        if response['AutoScalingGroups']:
            asg = response['AutoScalingGroups'][0]
            desired_capacity = asg['DesiredCapacity']
            print(f"Current server count: {desired_capacity}")
            return desired_capacity
        else:
            print(f"Auto Scaling Group '{AUTO_SCALING_GROUP_NAME}' not found. Using default.")
            return 2  # Default fallback
            
    except Exception as e:
        print(f"Error getting server count: {str(e)}")
        return 2  # Default fallback

def get_business_context():
    """
    Reads the business_calendar.json file from S3 to get the current 
    is_sale_active status.
    """
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=BUSINESS_CALENDAR_KEY
        )
        
        calendar_data = json.loads(response['Body'].read().decode('utf-8'))
        is_sale_active = calendar_data.get('is_sale_active', 0)
        print(f"Sale active status: {is_sale_active}")
        return is_sale_active
        
    except Exception as e:
        print(f"Error reading business calendar: {str(e)}")
        return 0  # Default to no sale

def get_historical_metrics():
    """
    Reads the last 100 rows from multi_metric_data.csv in S3 and formats 
    them for the dashboard.
    """
    try:
        response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=DATA_FILE_KEY
        )
        
        # Read CSV data
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))
        
        # Get last 100 rows
        df_recent = df.tail(100).copy()
        
        # Ensure timestamp column exists
        if 'timestamp' not in df_recent.columns:
            print("Warning: 'timestamp' column not found in data")
            return []
        
        # Format data for dashboard
        historical_data = []
        for _, row in df_recent.iterrows():
            data_point = {
                'timestamp': str(row['timestamp']),
                'cpu': round(float(row.get('cpu', row.get('cpu_utilization', 0))), 2),
                'network_in': round(float(row.get('network_in', 0)), 2),
                'request_count': round(float(row.get('request_count', 0)), 2),
                'is_sale_active': int(row.get('is_sale_active', 0)),
                'server_count_at_time': estimate_server_count(float(row.get('cpu', row.get('cpu_utilization', 0))))
            }
            historical_data.append(data_point)
        
        print(f"Retrieved {len(historical_data)} historical data points")
        return historical_data
        
    except Exception as e:
        print(f"Error reading historical metrics: {str(e)}")
        return []

def estimate_server_count(cpu_utilization):
    """
    Estimates the server count based on CPU utilization.
    This is a simple heuristic - in production, you'd track actual scaling events.
    """
    if cpu_utilization < 40:
        return 1
    elif cpu_utilization < 60:
        return 2
    elif cpu_utilization < 75:
        return 3
    else:
        return 4
