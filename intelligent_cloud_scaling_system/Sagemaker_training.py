import boto3
import sagemaker
from sagemaker.estimator import Estimator

# --- Configuration ---
# ======================== This line has been updated ========================
# 1. Paste the SageMaker Role ARN you created in the guide.
SAGEMAKER_ROLE_ARN = "arn:aws:iam::780139019966:role/MySageMakerRole" # IMPORTANT: Replace with your actual SageMaker Role ARN
# ==============================================================================

# --- This has been updated to match your AWS region ---
S3_BUCKET_NAME = "intelligent-scaling-demo-sathvik"
AWS_REGION = "ap-south-1" # Changed to ap-south-1 to match S3 bucket region

def main():
    """
    This script starts a training job on Amazon SageMaker.
    It does NOT run TensorFlow on your local machine.
    """
    print("--- Initializing SageMaker Session ---")
    
    if "YOUR_ACCOUNT_ID" in SAGEMAKER_ROLE_ARN:
        print("❌ CRITICAL ERROR: Please update the SAGEMAKER_ROLE_ARN in this script with your AWS Account ID and SageMaker Execution Role name.")
        return

    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))

    # Define the S3 locations for our data and model
    s3_input_data_path = f"s3://{S3_BUCKET_NAME}/multi_metric_data.csv"
    s3_output_model_path = f"s3://{S3_BUCKET_NAME}/"

    print(f"Data will be read from: {s3_input_data_path}")
    print(f"Model will be saved to: {s3_output_model_path}")

    # Configure the training job
    # This tells SageMaker to use a pre-built environment with TensorFlow
    estimator = Estimator(
        image_uri=sagemaker.image_uris.retrieve("tensorflow", AWS_REGION, "2.11.0", image_scope="training"), # Changed TensorFlow version to 2.11.0 (without -gpu-py39)
        role=SAGEMAKER_ROLE_ARN,
        instance_count=1,
        instance_type="ml.t3.medium", # A cost-effective instance type for training
        sagemaker_session=sagemaker_session,
        output_path=s3_output_model_path,
        entry_point="train_model.py", # The script SageMaker will run
        source_dir="./intelligent_cloud_scaling_system" # The folder containing the script
    )

    print("\n--- Starting SageMaker Training Job ---")
    print("This will take a few minutes. You can monitor the progress in the AWS SageMaker console.")
    
    # Start the job. This call is non-blocking.
    estimator.fit({"training": s3_input_data_path})

    print("\n✅ Training job successfully launched!")
    print("The trained model (lstm_model.h5) will appear in your S3 bucket when complete.")

if __name__ == "__main__":
    main()
