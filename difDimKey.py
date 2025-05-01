import os
import pandas as pd

# Base directory
base_dir = "/home/zhaohuan/桌面/keydim-SimpleLab"

# Metrics to compare
metrics = ["RMSE", "MAE", "MAPE"]

# Store results for A-C and C-A
results_ac = []
results_ca = []

# Traverse directories
for root, dirs, files in os.walk(base_dir):
    if "metrics.csv" in files:
        metrics_path = os.path.join(root, "metrics.csv")
        relative_path = os.path.relpath(root, base_dir)
        keydim_combination = relative_path.split(os.sep)[0]
        direction = relative_path.split(os.sep)[-1]  # Get the last part (A-C or C-A)
        
        # Read metrics.csv
        try:
            df = pd.read_csv(metrics_path)
            row = df.iloc[0]  # Assuming the first row contains the relevant data
            result = {
                "combine": keydim_combination,
                "RMSE": row["RMSE"],
                "MAE": row["MAE"],
                "MAPE": row["MAPE"]
            }
            if direction == "A-C":
                results_ac.append(result)
            elif direction == "C-A":
                results_ca.append(result)
        except Exception as e:
            print(f"Error reading {metrics_path}: {e}")

# Save results to CSV
output_ac = pd.DataFrame(results_ac)
output_ca = pd.DataFrame(results_ca)

output_ac.to_csv("metrics_summary_A-C.csv", index=False)
output_ca.to_csv("metrics_summary_C-A.csv", index=False)

print("Metrics summary saved for A-C and C-A directions.")