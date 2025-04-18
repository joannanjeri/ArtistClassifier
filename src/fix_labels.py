import pandas as pd

label_map = {
    'Albrecht_Du╠êrer': 'Albrecht_Durer',
    'Albrecht_DuΓòá├¬rer': 'Albrecht_Durer',
}

def clean_labels(file_path):
    df = pd.read_csv(file_path)
    df['artist'] = df['artist'].map(lambda x: label_map.get(x, x)) #replace labels if in the map
    print(f"Cleaned labels in {file_path}:")

for split in ['train', 'val', 'test']:
    clean_labels(f'data/{split}.csv')
    