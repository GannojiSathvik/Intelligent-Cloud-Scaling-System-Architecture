# API Setup Guide: Deploying the Backend

This guide will walk you through deploying the `GetDashboardDataFunction.py` on AWS Lambda and exposing it to the web using Amazon API Gateway. This will create the live API endpoint that your `index.html` dashboard needs to fetch data.

---

### Prerequisites

1.  **An AWS Account:** You'll need an active AWS account with permissions to create Lambda functions, S3 buckets, IAM roles, and API Gateway endpoints.
2.  **S3 Bucket:** You must have an S3 bucket (e.g., `my-intelligent-scaling-data-bucket`) that contains:
    *   `multi_metric_data.csv`
    *   `business_calendar.json`
3.  **Auto Scaling Group:** An active EC2 Auto Scaling Group that you want to monitor.

---

### Step 1: Create an IAM Role for the Lambda Function

Your Lambda function needs permission to access S3 and Auto Scaling. You'll create an IAM role that grants these permissions.

1.  Navigate to the **IAM** service in the AWS Console.
2.  Go to **Roles** and click **Create role**.
3.  For **Trusted entity type**, select **AWS service**.
4.  For **Use case**, choose **Lambda**.
5.  Click **Next**.
6.  On the **Add permissions** page, search for and add the following two policies:
    *   `AmazonS3ReadOnlyAccess` (Allows reading from your S3 bucket)
    *   `AutoScalingReadOnlyAccess` (Allows reading from your Auto Scaling Group)
7.  Click **Next**.
8.  Give the role a descriptive name, like `LambdaDashboardDataRole`, and click **Create role**.

---

### Step 2: Create the Lambda Function

Now you'll create the Lambda function and upload your Python code.

1.  Navigate to the **Lambda** service in the AWS Console.
2.  Click **Create function**.
3.  Select **Author from scratch**.
4.  **Function name:** Enter `GetDashboardDataFunction`.
5.  **Runtime:** Choose **Python 3.9** (or a newer Python version).
6.  **Architecture:** Keep the default `x86_64`.
7.  **Permissions:** Expand **Change default execution role**, select **Use an existing role**, and choose the `LambdaDashboardDataRole` you created in Step 1.
8.  Click **Create function**.

---

### Step 3: Configure the Lambda Function and Upload Code

1.  **Upload the Code:**
    *   Because our script has a dependency (`pandas`), we need to upload it as a `.zip` file.
    *   On your local machine, create a new folder (e.g., `lambda_package`).
    *   Copy `GetDashboardDataFunction.py` into this folder.
    *   Install the `pandas` library into the same folder:
        ```bash
        pip install pandas -t ./
        ```
    *   Zip the *contents* of the `lambda_package` folder. Make sure not to zip the folder itself.
    *   Back in the Lambda console, in the **Code source** section, click **Upload from** and select **.zip file**. Upload the zip file you just created.

2.  **Update the Handler:**
    *   Go to the **Runtime settings** section and click **Edit**.
    *   Change the **Handler** to `GetDashboardDataFunction.lambda_handler`.
    *   Click **Save**.

3.  **Increase Timeout:**
    *   Go to the **Configuration** tab and select **General configuration**.
    *   Click **Edit**.
    *   Set the **Timeout** to **15 seconds** to give the function enough time to fetch data.
    *   Click **Save**.

---

### Step 4: Create the HTTP API Gateway

This will create a public URL that triggers your Lambda function.

1.  Navigate to the **API Gateway** service in the AWS Console.
2.  Find the box for **HTTP API** and click **Build**.
3.  Click **Add integration**.
4.  Select **Lambda** from the dropdown.
5.  For **Lambda function**, choose `GetDashboardDataFunction`.
6.  Give your API a name, like `IntelligentScalingDashboardAPI`.
7.  Click **Next**.

8.  **Configure Routes:**
    *   **Method:** Select `GET`.
    *   **Resource path:** Enter `/dashboard`.
    *   The **Integration target** should already be set to your Lambda function.
    *   Click **Next**.

9.  **Define Stages:**
    *   Keep the default `$default` stage settings.
    *   Click **Next**.

10. Review and click **Create**.

---

### Step 5: Get the API Invoke URL

Once the API is created, you need its public URL.

1.  In the API Gateway console, with your new API selected, you will see an **Invoke URL** on the main details page.
2.  It will look something like this: `https://abcdef123.execute-api.us-east-1.amazonaws.com`.
3.  **Important:** You need to append your route to this URL. The final URL you need is the Invoke URL + your resource path (`/dashboard`).

    **Final URL Example:** `https://abcdef123.execute-api.us-east-1.amazonaws.com/dashboard`

---

### Step 6: Update the `index.html` File

1.  Open your local `index.html` file.
2.  Find this line of JavaScript:
    ```javascript
    const API_ENDPOINT = 'YOUR_API_GATEWAY_INVOKE_URL_HERE';
    ```
3.  Replace `YOUR_API_GATEWAY_INVOKE_URL_HERE` with the final URL you obtained in the previous step.
4.  Save the file.

**You are now done!** Open the `index.html` file in your web browser to see your live dashboard.
