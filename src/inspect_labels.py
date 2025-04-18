import pandas as pd

for split in ['train', 'val', 'test']:
    df = pd.read_csv(f'data/{split}.csv')
    print(f"\nUnique artists in {split}.csv:")
    print(sorted(df['artist'].unique()))

