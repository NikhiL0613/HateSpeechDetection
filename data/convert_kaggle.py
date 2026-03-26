import pandas as pd
import os

print("=" * 60)
print("  CONVERTING KAGGLE DATASET")
print("=" * 60)

input_path = "data/labeled_data.csv"

if not os.path.exists(input_path):
    print(f"ERROR: File not found: {input_path}")
    exit()

df = pd.read_csv(input_path)
print(f"Loaded {len(df):,} samples")

df["label"] = df["class"].apply(lambda x: 1 if x in [0, 1] else 0)
df["text"] = df["tweet"]
output_df = df[["text", "label"]].dropna()
output_df = output_df[output_df["text"].str.strip() != ""].reset_index(drop=True)

print(f"Hate:     {(output_df['label'] == 1).sum():,}")
print(f"Non-hate: {(output_df['label'] == 0).sum():,}")

output_df.to_csv("data/dataset.csv", index=False)
print("Saved to data/dataset.csv")
