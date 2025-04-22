import pandas as pd
import os
import shutil
import stat

#corrupted artist names
corrupted_artists = [
    "Albrecht_Du╠êrer",
    "Albrecht_DuΓòá├¬rer"
]

csv_files = [
    'data/train.csv',
    'data/val.csv',
    'data/test.csv',
    'data/train_top15.csv',
    'data/val_top15.csv',
    'data/test_top15.csv'
]

def clean_csv(csv_path, corrupted):
    if not os.path.exists(csv_path):
        print(f" File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    before = len(df)

    #remove rows where artist is in corrupted list
    df = df[~df['artist'].isin(corrupted)]
    after = len(df)

    df.to_csv(csv_path, index=False)
    print(f" Cleaned {csv_path}: Removed {before - after} rows")

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

for csv in csv_files:
    clean_csv(csv, corrupted_artists)

corrupted_folders = [
    'data/images/Albrecht_Du╠êrer',
    'data/images/Albrecht_DuΓòá├¬rer'
]

for folder in corrupted_folders:
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder, onerror=on_rm_error)
            print(f" Removed folder: {folder}")
        except Exception as e:
            print(f" Error removing folder {folder}: {e}")
    else:
        print(f" Folder not found: {folder}")
        