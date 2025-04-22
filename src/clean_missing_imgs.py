import pandas as pd
import os

def clean_missing_images(csv_path):
    ''' Clean the CSV file by removing rows with missing images '''

    df = pd.read_csv(csv_path)
    og_len = len(df)

    #keep only rows where image file exists
    df = df[df['file_path'].apply(lambda path: os.path.exists(path))]

    removed = og_len - len(df)
    df.to_csv(csv_path, index=False)
    print(f" Cleaned {csv_path}: Removed {removed} missing images")

if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        path = f"data/{split}_top15.csv"
        if os.path.exists(path):
            clean_missing_images(path)
        else:
            print(f" File not found: {path}")

            