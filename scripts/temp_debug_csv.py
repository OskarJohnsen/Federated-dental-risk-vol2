"""Debug script to inspect CSV columns."""
import pandas as pd
from pathlib import Path
from digit_fr.core.paths import root_path

csv_path = root_path('data', 'results', 'A', 'non-iid', 'sweep_beta', 'wandb_export_beta_runs.csv')

if not csv_path.exists():
    print(f"CSV not found: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} runs")
print(f"Total columns: {len(df.columns)}\n")

# Find metric columns
metric_cols = [c for c in df.columns if any(x in c.lower() for x in ['category', 'mse', 'fleiss', 'accuracy', 'f1'])]
print(f"Metric columns ({len(metric_cols)}):")
for col in sorted(metric_cols):
    non_null = df[col].notna().sum()
    if non_null > 0:
        sample_val = df[col].dropna().iloc[0]
        print(f"  {col}: {non_null} non-null (sample: {sample_val})")

# Check for specific patterns
print("\n\nChecking for specific patterns:")
patterns = [
    'test/category_per_client_f1_macro_risk_AlveolarOsteitis',
    'summary_test/category_per_client_f1_macro_risk_AlveolarOsteitis',
    'summary_test_category_per_client_f1_macro_risk_AlveolarOsteitis',
    'category_per_client_f1_macro_risk_AlveolarOsteitis',
]

for pattern in patterns:
    matches = [c for c in df.columns if pattern in c]
    if matches:
        print(f"  ✓ Found: {pattern}")
        for m in matches:
            non_null = df[m].notna().sum()
            print(f"      -> {m}: {non_null} non-null")
    else:
        print(f"  ✗ Not found: {pattern}")
