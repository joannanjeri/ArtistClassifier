import pandas as pd

label_map = {
    'Albrecht_Du╠êrer': 'Albrecht_Durer',
    'Albrecht_DuΓòá├¬rer': 'Albrecht_Durer',
}

def clean_labels(file_path):
    df = pd.read_csv(file_path)
    
    if 'artist' not in df.columns:
        print(f" 'artist' column not found in {file_path}")
        return
    
    #apply mapping to the 'artist' column
    df['artist'] = df['artist'].map(lambda x: label_map.get(x, x))

    #overwrite the csv file with the cleaned labels
    df.to_csv(file_path, index=False)
    print(f"Cleaned labels in {file_path}")

#apply the cleaning function to all csv files
for split in ['train', 'val', 'test']:
    clean_labels(f'data/{split}.csv')

    