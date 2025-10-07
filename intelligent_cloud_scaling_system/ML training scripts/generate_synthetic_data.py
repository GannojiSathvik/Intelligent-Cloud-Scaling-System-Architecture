import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration
DATA_POINTS = 15 * 24 * 12  # 15 days, 12 points per hour (every 5 mins)
START_DATE = datetime.now() - timedelta(days=15)

# --- Create Timestamps ---
timestamps = [START_DATE + timedelta(minutes=5 * i) for i in range(DATA_POINTS)]

# --- Simulate Daily Cyclical Patterns ---
time_of_day_factor = np.sin(np.linspace(0, 15 * 2 * np.pi, DATA_POINTS))  # 15 full cycles for 15 days

# Base metrics with cyclical pattern and noise
base_cpu = 20 + 15 * time_of_day_factor + np.random.normal(0, 2, DATA_POINTS)
base_network = 100 + 80 * time_of_day_factor + np.random.normal(0, 10, DATA_POINTS)
base_requests = 500 + 400 * time_of_day_factor + np.random.normal(0, 50, DATA_POINTS)

# --- Introduce "Sale Day" Events (Context-Awareness) ---
is_sale_active = np.zeros(DATA_POINTS)
# Let's assume sales happen on the 5th, 10th, and 15th day
sale_days = [5, 10, 15]
for day in sale_days:
    start_index = (day - 1) * 24 * 12
    end_index = day * 24 * 12
    is_sale_active[start_index:end_index] = 1

# Amplify metrics on sale days
sale_cpu_boost = 40 * is_sale_active
sale_network_boost = 300 * is_sale_active
sale_requests_boost = 2000 * is_sale_active

final_cpu = np.clip(base_cpu + sale_cpu_boost + np.random.normal(0, 5, DATA_POINTS), 5, 95)
final_network = np.clip(base_network + sale_network_boost + np.random.normal(0, 20, DATA_POINTS), 20, 1000)
final_requests = np.clip(base_requests + sale_requests_boost + np.random.normal(0, 100, DATA_POINTS), 100, 5000)

# --- Assemble DataFrame ---
data = {
    'timestamp': timestamps,
    'cpu_utilization': final_cpu,
    'network_in': final_network,
    'request_count': final_requests,
    'is_sale_active': is_sale_active
}
df = pd.DataFrame(data)

# --- Save to CSV ---
output_path = 'multi_metric_data.csv'
df.to_csv(output_path, index=False)

print(f"Successfully generated {len(df)} data points.")
print(f"Data saved to '{output_path}'.")
print("\nSample of the generated data:")
print(df.head())
print("\nData summary:")
print(df.describe())
