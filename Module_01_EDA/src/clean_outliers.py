import os
import pandas as pd


# Input and output paths
input_path = os.path.join("data", "raw_business_data.csv")
output_path = os.path.join("data", "cleaned_business_data.csv")


# Load the raw data
df = pd.read_csv(input_path)

print(f"Original rows: {len(df)}")


# Apply Tukey 1.5 × IQR rule to numeric columns
numeric_columns = df.select_dtypes(include="number").columns

keep_rows = pd.Series(True, index=df.index)

for column in numeric_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    keep_rows &= df[column].between(lower_bound, upper_bound)


# Create cleaned dataset
cleaned_df = df[keep_rows].copy()

cleaned_df.to_csv(output_path, index=False)

print(f"Cleaned rows: {len(cleaned_df)}")
print(f"Rows removed: {len(df) - len(cleaned_df)}")
print(f"[+] Saved cleaned data to: {output_path}")