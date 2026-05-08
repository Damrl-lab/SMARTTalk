import pandas as pd
import numpy as np

# Input and output file names (change if needed)
input_file = "mb1_mb2_status_comparison.xlsx"
output_file = "mb1_mb2_status_comparison_with_f05.xlsx"

# Read the Excel sheet
df = pd.read_excel(input_file)

# Define F-beta (here beta = 0.5) function
def f_beta(p, r, beta=0.5):
    beta2 = beta ** 2
    denom = beta2 * p + r
    # Avoid division by zero: if denom == 0, set F to 0
    f = np.where(
        denom == 0,
        0.0,
        (1 + beta2) * p * r / denom
    )
    return f

# Compute F0.5 for MB1 and MB2
df["MB1_F05"] = f_beta(df["MB1_P"], df["MB1_R"], beta=0.5)
df["MB2_F05"] = f_beta(df["MB2_P"], df["MB2_R"], beta=0.5)

# Round to two decimal places
df["MB1_F05"] = df["MB1_F05"].round(2)
df["MB2_F05"] = df["MB2_F05"].round(2)

# Save to a new Excel file
df.to_excel(output_file, index=False)

print(f"Saved updated file with F0.5 scores to: {output_file}")
