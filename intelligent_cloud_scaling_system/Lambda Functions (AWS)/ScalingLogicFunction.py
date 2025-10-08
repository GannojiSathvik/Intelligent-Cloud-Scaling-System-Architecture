import json
import boto3
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import csv
from io import StringIO

# --- Configuration ---
S3_BUCKET_NAME = "my-intelligent-scaling-data-bucket"
DATA_FILE_KEY = "multi_metric_data.csv"
MODEL_FILE_KEY = "models/lstm_model_optimized.pth" # Corrected path from README
# Allow overriding the ASG name via environment variable for flexibility in different environments
AUTO_SCALING_GROUP_NAME = os.getenv("AUTO_SCALING_GROUP_NAME", "intelligent-scaling-demo-sathvik")
LOCAL_MODEL_PATH = "/tmp/lstm_model.pth"

# Scaling Thresholds
CPU_UPPER_THRESHOLD = 70.0
CPU_LOWER_THRESHOLD = 35.0

s3_client = boto3.client('s3')
autoscaling_client = boto3.client('autoscaling')

# Define the same LSTM model architecture used in training
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def lambda_handler(event, context):
    """
    This Lambda function predicts future CPU load using a trained LSTM model
    and adjusts the Desired Capacity of an Auto Scaling Group.
    """
    try:
        # --- 1. Download Model from S3 ---
        print(f"Downloading model '{MODEL_FILE_KEY}' from S3 bucket '{S3_BUCKET_NAME}'...")
        s3_client.download_file(S3_BUCKET_NAME, MODEL_FILE_KEY, LOCAL_MODEL_PATH)
        print("Model downloaded successfully.")

        # --- 2. Fetch the Latest Data Sequence from S3 ---
        print(f"Fetching data from '{DATA_FILE_KEY}'...")
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=DATA_FILE_KEY)
        content = obj['Body'].read().decode('utf-8')
        
        df = pd.read_csv(StringIO(content))
        
        if len(df) < 12:
            return {
                'statusCode': 200,
                'body': json.dumps('Not enough data to create a sequence (need at least 12 data points).')
            }
        
        latest_data = df.tail(12)
        print(f"Successfully fetched {len(latest_data)} latest data rows.")

        # --- 3. Preprocess the data ---
        features = ['cpu_utilization', 'network_in', 'request_count', 'is_sale_active']
        sequence_data = latest_data[features].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df[features].values)
        scaled_sequence = scaler.transform(sequence_data)

        input_sequence = np.reshape(scaled_sequence, (1, 12, len(features)))
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32)

        # --- 4. Load Model and Make Prediction ---
        input_size = 4
        hidden_size = 50
        num_layers = 2
        output_size = 1
        
        model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        model.load_state_dict(torch.load(LOCAL_MODEL_PATH))
        model.eval()

        with torch.no_grad():
            predicted_scaled_cpu = model(input_tensor).item()

        dummy_array = np.zeros((1, len(features)))
        dummy_array[0, 0] = predicted_scaled_cpu
        predicted_cpu = scaler.inverse_transform(dummy_array)[0, 0]
        
        print(f"Model prediction: CPU Utilization = {predicted_cpu:.2f}%")

        # --- 5. Implement Scaling Logic ---
        asg_resp = autoscaling_client.describe_auto_scaling_groups(
            AutoScalingGroupNames=[AUTO_SCALING_GROUP_NAME]
        )
        if not asg_resp.get('AutoScalingGroups'):
            msg = f"ASG '{AUTO_SCALING_GROUP_NAME}' not found. Predicted CPU was {predicted_cpu:.2f}%. No scaling action attempted."
            print(msg)
            return {'statusCode': 200, 'body': json.dumps(msg)}

        asg = asg_resp['AutoScalingGroups'][0]
        current_capacity = asg['DesiredCapacity']
        max_capacity = asg['MaxSize']
        min_capacity = asg['MinSize']
        print(f"ASG State: Desired={current_capacity}, Min={min_capacity}, Max={max_capacity}")

        new_capacity = current_capacity

        if predicted_cpu > CPU_UPPER_THRESHOLD:
            new_capacity = min(current_capacity + 1, max_capacity)
            if new_capacity > current_capacity:
                print(f"SCALING UP: Prediction ({predicted_cpu:.2f}%) is above threshold ({CPU_UPPER_THRESHOLD}%).")
            else:
                print("ACTION BLOCKED: Prediction is high, but ASG is already at max capacity.")
        elif predicted_cpu < CPU_LOWER_THRESHOLD:
            new_capacity = max(current_capacity - 1, min_capacity)
            if new_capacity < current_capacity:
                print(f"SCALING DOWN: Prediction ({predicted_cpu:.2f}%) is below threshold ({CPU_LOWER_THRESHOLD}%).")
            else:
                print("ACTION BLOCKED: Prediction is low, but ASG is already at min capacity.")
        else:
            print("NO ACTION: Predicted CPU is within normal operating thresholds.")

        if new_capacity != current_capacity:
            autoscaling_client.set_desired_capacity(
                AutoScalingGroupName=AUTO_SCALING_GROUP_NAME,
                DesiredCapacity=new_capacity,
                HonorCooldown=False
            )
            print(f"Set DesiredCapacity to {new_capacity}.")

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
        if os.path.exists(LOCAL_MODEL_PATH):
            os.remove(LOCAL_MODEL_PATH)
