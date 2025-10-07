import json
import boto3
import os

# --- Configuration ---
S3_BUCKET_NAME = "my-intelligent-scaling-data-bucket"
DATA_FILE_KEY = "multi_metric_data.csv"
MODEL_FILE_KEY = "lstm_model.pth"
AUTO_SCALING_GROUP_NAME = "intelligent-scaling-demo-sathvik"
LOCAL_MODEL_PATH = "/tmp/lstm_model.pth"

# Scaling Thresholds
CPU_UPPER_THRESHOLD = 70.0  # Scale up if predicted CPU > 70%
CPU_LOWER_THRESHOLD = 35.0  # Scale down if predicted CPU < 35%

s3_client = boto3.client('s3')
autoscaling_client = boto3.client('autoscaling')

def lambda_handler(event, context):
    """
    This Lambda function predicts future CPU load using a trained LSTM model
    and adjusts the Desired Capacity of an Auto Scaling Group.
    """
    try:
        # --- 1. Model path disabled in this environment ---
        # We intentionally use a lightweight heuristic (no NumPy/Torch/Sklearn) for Lambda demo
        print("Using heuristic prediction path (no external ML libraries).")

        # --- 2. Fetch the Latest Data Sequence from S3 ---
        # Use a pandas-free path to avoid dependency when falling back
        import csv
        from io import StringIO
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=DATA_FILE_KEY)
            content = obj['Body'].read().decode('utf-8')
            rows = [r for r in csv.reader(StringIO(content))]
            # Expect header in first row
            header = rows[0] if rows else []
            data_rows = rows[1:] if len(rows) > 1 else []
            # Filter malformed rows (need at least 5 cols)
            data_rows = [r for r in data_rows if isinstance(r, list) and len(r) >= 5]
            if len(data_rows) < 1:
                return {
                    'statusCode': 200,
                    'body': json.dumps('No valid data rows available yet. Skipping scaling.')
                }
            # Use up to last 12 rows
            latest_rows = data_rows[-12:] if len(data_rows) >= 12 else data_rows
            print(f"Successfully fetched {len(latest_rows)} latest data rows.")
        except Exception as e:
            return f"Error: Failed to read data from S3. {e}"

        # --- 3. Prepare data for heuristic ---
        features = ['cpu_utilization', 'network_in', 'request_count', 'is_sale_active']
        # Column indices based on expected CSV: timestamp,cpu_utilization,network_in,request_count,is_sale_active
        CPU_COL = 1
        NET_COL = 2
        REQ_COL = 3
        SALE_COL = 4

        # --- 4. Compute heuristic prediction ---
        try:
            cpu_values = []
            for r in latest_rows:
                try:
                    cpu_values.append(float(r[CPU_COL]))
                except Exception:
                    continue
            if not cpu_values:
                return {
                    'statusCode': 200,
                    'body': json.dumps('No numeric CPU values available for heuristic. Skipping scaling.')
                }
            predicted_cpu = sum(cpu_values) / len(cpu_values)
            print(f"Heuristic prediction (avg of {len(cpu_values)} rows): CPU Utilization = {predicted_cpu:.2f}%")
        except Exception as e:
            return f"Error: Failed heuristic computation: {e}"

        # --- 5. Implement Scaling Logic ---
        asg_resp = autoscaling_client.describe_auto_scaling_groups(
            AutoScalingGroupNames=[AUTO_SCALING_GROUP_NAME]
        )
        if not asg_resp.get('AutoScalingGroups'):
            msg = (
                f"ASG '{AUTO_SCALING_GROUP_NAME}' not found. Predicted CPU was "
                f"{predicted_cpu:.2f}%. No scaling action attempted."
            )
            print(msg)
            return {
                'statusCode': 200,
                'body': json.dumps(msg)
            }

        asg_description = asg_resp['AutoScalingGroups'][0]

        current_capacity = asg_description['DesiredCapacity']
        max_capacity = asg_description['MaxSize']
        min_capacity = asg_description['MinSize']
        print(f"ASG State: Desired={current_capacity}, Min={min_capacity}, Max={max_capacity}")

        new_capacity = current_capacity

        if predicted_cpu > CPU_UPPER_THRESHOLD:
            new_capacity = min(current_capacity + 1, max_capacity)
            if new_capacity > current_capacity:
                print(f"SCALING UP: Prediction ({predicted_cpu:.2f}%) is above threshold ({CPU_UPPER_THRESHOLD}%). Setting DesiredCapacity to {new_capacity}.")
            else:
                print("ACTION BLOCKED: Prediction is high, but ASG is already at max capacity.")
        elif predicted_cpu < CPU_LOWER_THRESHOLD:
            new_capacity = max(current_capacity - 1, min_capacity)
            if new_capacity < current_capacity:
                print(f"SCALING DOWN: Prediction ({predicted_cpu:.2f}%) is below threshold ({CPU_LOWER_THRESHOLD}%). Setting DesiredCapacity to {new_capacity}.")
            else:
                print("ACTION BLOCKED: Prediction is low, but ASG is already at min capacity.")
        else:
            print("NO ACTION: Predicted CPU is within normal operating thresholds.")

        if new_capacity != current_capacity:
            autoscaling_client.set_desired_capacity(
                AutoScalingGroupName=AUTO_SCALING_GROUP_NAME,
                DesiredCapacity=new_capacity,
                HonorCooldown=False # Set to True in production to avoid flapping
            )

        return {
            'statusCode': 200,
            'body': json.dumps(f'Scaling logic executed. Predicted CPU: {predicted_cpu:.2f}%. New capacity: {new_capacity}')
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error during scaling logic: {str(e)}')
        }
    finally:
        # Clean up the downloaded model file from the /tmp/ directory
        if os.path.exists(LOCAL_MODEL_PATH):
            os.remove(LOCAL_MODEL_PATH)
