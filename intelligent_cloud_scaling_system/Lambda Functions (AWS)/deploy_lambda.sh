#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# Get the directory where the script is located to build robust paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

FUNCTION_NAME="ScalingLogicFunction"
HANDLER_FILE="$SCRIPT_DIR/ScalingLogicFunction.py"
ZIP_FILE="$SCRIPT_DIR/scaling_logic_deployment.zip"
REQUIREMENTS_FILE="$SCRIPT_DIR/lambda_requirements.txt"
PACKAGE_DIR="$SCRIPT_DIR/package"

# --- 1. Check for AWS CLI ---
if ! command -v aws &> /dev/null
then
    echo "AWS CLI could not be found. Please install and configure it."
    exit 1
fi

# --- 2. Create a clean package directory ---
echo "Creating a clean package directory at $PACKAGE_DIR..."
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

# --- 3. Install dependencies ---
echo "Installing dependencies from $REQUIREMENTS_FILE into $PACKAGE_DIR..."
# Using python -m pip to avoid issues with venv pathing
python -m pip install --target "$PACKAGE_DIR" -r "$REQUIREMENTS_FILE"

# --- 4. Add Lambda function code to the package ---
echo "Adding Lambda function code from $HANDLER_FILE..."
cp "$HANDLER_FILE" "$PACKAGE_DIR/"

# --- 5. Create the deployment package (zip file) ---
echo "Creating deployment package: $ZIP_FILE..."
(cd "$PACKAGE_DIR" && zip -r9 "$ZIP_FILE" .)

# --- 6. Deploy to AWS Lambda ---
echo "Deploying to AWS Lambda function '$FUNCTION_NAME'..."
aws lambda update-function-code \
    --function-name "$FUNCTION_NAME" \
    --zip-file "fileb://$ZIP_FILE"

# --- 7. Clean up ---
echo "Cleaning up temporary files..."
rm -rf "$PACKAGE_DIR"
rm "$ZIP_FILE"

echo "Deployment script finished successfully."
