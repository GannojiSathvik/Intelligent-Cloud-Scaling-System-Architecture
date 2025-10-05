# Intelligent Cloud Scaling System

This project demonstrates a sophisticated, predictive, and context-aware system for automatically scaling cloud resources on AWS.

## How It Works: A Three-Stage Pipeline

The system operates in a continuous, automated loop with three distinct stages:

1.  **Data Collection (`DataCollectorFunction.py`)**:
    *   Every 5 minutes, an AWS Lambda function gathers real-time server metrics from Amazon CloudWatch, specifically `CPUUtilization` and `NetworkIn`.
    *   Crucially, it also reads a `business_calendar.json` file from an S3 bucket to check for external business events (e.g., `is_sale_active`).
    *   This combined data is appended as a new row to a growing historical dataset (`multi_metric_data.csv`) in S3.

2.  **Model Training (`train_model.py`)**:
    *   This Python script is run periodically (e.g., once a day) to train our intelligent model.
    *   It downloads the entire historical dataset from S3.
    *   It uses this data to train a Long Short-Term Memory (LSTM) neural network, which is excellent at learning from time-series data.
    *   The result is a trained model file (`lstm_model.h5`) that understands the complex patterns in your server load. This file is then uploaded back to S3.

3.  **Predictive Scaling (`ScalingLogicFunction.py`)**:
    *   Every 5 minutes, a second Lambda function executes the core logic.
    *   It downloads the latest trained model from S3.
    *   It fetches the most recent sequence of data (e.g., the last 60 minutes).
    *   It feeds this data into the model to **predict the server load for the next 5 minutes**.
    *   Based on this prediction, it makes an intelligent decision:
        *   If the predicted load is high (e.g., above 70%), it proactively adds a new EC2 instance to the Auto Scaling Group **before** the traffic hits.
        *   If the predicted load is low (e.g., below 35%), it removes an instance to save costs.

## Novelty: What Makes This System Different?

This system is a significant improvement over standard cloud auto-scaling in three key ways:

1.  **Predictive, Not Reactive**:
    *   **Standard Scaling**: Waits for CPU to get high, *then* reacts. This lag can lead to slow performance for users during traffic spikes.
    *   **This System**: **Forecasts** future load and scales up *in advance*. It prevents performance issues before they happen, ensuring a smooth user experience.

2.  **Multi-variate Analysis**:
    *   **Standard Scaling**: Typically only looks at one metric, like CPU utilization. This can be misleading.
    *   **This System**: Considers multiple data streams simultaneously (`cpu_utilization`, `network_in`, `request_count`). This provides a more holistic and accurate understanding of the true server load.

3.  **Context-Awareness (The Core Innovation)**:
    *   **Standard Scaling**: Is blind to the outside world. It cannot distinguish between a random traffic surge and a planned, high-traffic event like a marketing promotion.
    *   **This System**: Is **context-aware**. By incorporating the `is_sale_active` flag, the model learns the specific patterns associated with predictable business events. It understands that a "sale day" has a different traffic profile and can make much more aggressive and accurate scaling decisions, ensuring the system is perfectly prepared for predictable peaks.

In short, this system moves beyond simple, reactive scaling to an intelligent, predictive, and business-aware approach that optimizes for both performance and cost.
