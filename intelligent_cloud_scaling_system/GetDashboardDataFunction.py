import boto3
import json
import pandas as pd
import io
from datetime import datetime

# --- Configuration ---
S3_BUCKET_NAME = "sathvik-scaling-data-us-east-2"
AWS_REGION = "us-east-2" 
ASG_NAME = 'IntelligentScaling-ASG' 

DATA_FILE_KEY = "multi_metric_data.csv"
CALENDAR_FILE_KEY = "config/business_calendar.json"
SCALING_LOGIC_LOG_GROUP = '/aws/lambda/ScalingLogicFunction'

s3_client = boto3.client('s3', region_name=AWS_REGION)
autoscaling_client = boto3.client('autoscaling', region_name=AWS_REGION)
logs_client = boto3.client('logs', region_name=AWS_REGION)

def get_latest_prediction():
    """Scans CloudWatch Logs to find the last prediction message."""
    try:
        streams = logs_client.describe_log_streams(
            logGroupName=SCALING_LOGIC_LOG_GROUP,
            orderBy='LastEventTime',
            descending=True,
            limit=1
        )
        if not streams.get('logStreams'):
            return "No prediction yet"

        log_stream_name = streams['logStreams'][0]['logStreamName']
        events = logs_client.get_log_events(
            logGroupName=SCALING_LOGIC_LOG_GROUP,
            logStreamName=log_stream_name,
            limit=50,
            startFromHead=False
        )
        for event in reversed(events.get('events', [])):
            if "Model predicts future CPU" in event['message']:
                prediction = event['message'].split('will be: ')[1].split('%')[0]
                return f"{float(prediction):.2f}%"
        return "Not found in logs"
    except logs_client.exceptions.ResourceNotFoundException:
        return "Prediction service inactive"
    except Exception as e:
        print(f"Log read error: {e}")
        return "Log read error"

def get_scaling_activities():
    """Gets the last 5 scaling activities from the Auto Scaling Group."""
    try:
        response = autoscaling_client.describe_scaling_activities(
            AutoScalingGroupName=ASG_NAME,
            MaxRecords=5
        )
        activities = []
        for act in response.get('Activities', []):
            activities.append({
                "time": act['StartTime'].strftime('%I:%M %p'),
                "description": act['Description']
            })
        return activities
    except Exception as e:
        print(f"Scaling activity error: {e}")
        return []


def lambda_handler(event, context):
    # Handle preflight OPTIONS request for CORS
    if event.get('httpMethod') == 'OPTIONS' or event.get('requestContext', {}).get('http', {}).get('method') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Methods': 'GET,OPTIONS'
            },
            'body': ''
        }
        
    try:
        # --- Get Core Data ---
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=DATA_FILE_KEY)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        
        latest_prediction = get_latest_prediction()
        scaling_events = get_scaling_activities()
        
        # FINAL FIX: The error shows the column is not named 'cpu'.
        # This code now robustly handles either 'cpu' or 'cpu_utilization'
        # and renames it to 'cpu' for the dashboard.
        if 'cpu_utilization' in df.columns:
            historical_data = df.tail(100)[['timestamp', 'cpu_utilization']].rename(columns={'cpu_utilization': 'cpu'}).to_dict('records')
        elif 'cpu' in df.columns:
            historical_data = df.tail(100)[['timestamp', 'cpu']].to_dict('records')
        else:
            raise KeyError("Neither 'cpu' nor 'cpu_utilization' column found in the CSV file.")

        try:
            cal_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=CALENDAR_FILE_KEY)
            is_sale_active = json.loads(cal_obj['Body'].read().decode('utf-8')).get('is_sale_active', 0)
        except s3_client.exceptions.NoSuchKey:
            is_sale_active = 0

        response = autoscaling_client.describe_auto_scaling_groups(AutoScalingGroupNames=[ASG_NAME])
        current_server_count = response['AutoScalingGroups'][0]['DesiredCapacity']

        dashboard_data = {
            "current_server_count": current_server_count,
            "is_sale_active": is_sale_active,
            "historical_data": historical_data,
            "latest_prediction": latest_prediction,
            "scaling_events": scaling_events,
            "last_updated": datetime.now().strftime('%I:%M:%S %p')
        }

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            },
            'body': json.dumps(dashboard_data, default=str)
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET,OPTIONS'
            },
            'body': json.dumps({'error': str(e)})
        }
