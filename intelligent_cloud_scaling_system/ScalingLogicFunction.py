import json
import boto3
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# --- Configuration ---
S3_BUCKET_NAME = "my-intelligent-scaling-data-bucket"
DATA_FILE_KEY = "multi_metric_data.csv"
MODEL_FILE_KEY = "lstm_model.h5"
AUTO_SCALING_GROUP_NAME = "intelligent-scaling-demo-sathvik"
LOCAL_MODEL_PATH = "/tmp/lstm_model.h5"

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
        # --- 1. Download and Load the Trained Model ---
        try:
            s3_client.download_file(S3_BUCKET_NAME, MODEL_FILE_KEY, LOCAL_MODEL_PATH)
            model = load_model(LOCAL_MODEL_PATH)
            print(f"Successfully downloaded and loaded model '{MODEL_FILE_KEY}'.")
        except Exception as e:
            return f"Error: Failed to load model from S3. {e}"

        # --- 2. Fetch the Latest Data Sequence from S3 ---
        try:
            df = pd.read_csv(f"s3://{S3_BUCKET_NAME}/{DATA_FILE_KEY}")
            if len(df) < 12:
                return "Error: Not enough data to create a sequence (need at least 12 data points)."
            latest_data = df.tail(12) # Get the last 60 minutes of data
            print("Successfully fetched latest data sequence.")
        except Exception as e:
            return f"Error: Failed to read data from S3. {e}"

        # --- 3. Preprocess the Data for Prediction ---
        features = ['cpu_utilization', 'network_in', 'request_count', 'is_sale_active']
        sequence_data = latest_data[features].values

        # We must use the same scaling parameters as during training
        # For simplicity, we fit a new scaler here. In a robust system, you'd save/load the scaler.
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df[features].values) # Fit on the whole dataset to get the range
        scaled_sequence = scaler.transform(sequence_data)

        # Reshape for LSTM model [1, time_steps, features]
        input_sequence = np.reshape(scaled_sequence, (1, 12, len(features)))

        # --- 4. Make a Prediction ---
        predicted_scaled_cpu = model.predict(input_sequence)[0][0]

        # We need to inverse transform the prediction to get a real CPU value
        # Create a dummy array with the predicted value in the first column
        dummy_array = np.zeros((1, len(features)))
        dummy_array[0, 0] = predicted_scaled_cpu
        predicted_cpu = scaler.inverse_transform(dummy_array)[0, 0]

        print(f"Model prediction (next 5 mins): CPU Utilization = {predicted_cpu:.2f}%")

        # --- 5. Implement Scaling Logic ---
        asg_description = autoscaling_client.describe_auto_scaling_groups(
            AutoScalingGroupNames=[AUTO_SCALING_GROUP_NAME]
        )['AutoScalingGroups'][0]

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
