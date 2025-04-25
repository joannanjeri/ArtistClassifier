import pandas as pd
import os

def clean_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if 'file_path' not in df.columns:
        print(f"❌ 'file_path' column not found in {csv_path}")
        return

    original_len = len(df)

    df['file_path'] = df['file_path'].apply(lambda p: p.replace('\\', '/'))

    df = df[df['file_path'].apply(lambda p: os.path.exists(p))]

    removed = original_len - len(df)
    df.to_csv(csv_path, index=False)
    print(f"✅ Cleaned {csv_path}: Removed {removed} missing image(s)")

csv_paths = [
    'data/train_top15.csv',
    'data/train_top15_balanced.csv',
    'data/val_top15.csv',
    'data/test_top15.csv',
]

for csv in csv_paths:
    clean_csv(csv)
