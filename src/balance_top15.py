import pandas as pd
from pathlib import Path

CSV_PATH = 'data/train_top15.csv'
OUTPUT_CSV_PATH = 'data/train_top15_balanced.csv'
SAMPLES_PER_ARTIST = 127
SEED = 50

def balance_dataset(input_csv, output_csv, samples_per_artist, seed=50):
    ''' Balance the dataset by sampling a fixed number of images per artist '''
    
    if not Path(input_csv).exists():
        print(f" File not found: {input_csv}")
        return
    
    df = pd.read_csv(input_csv)

    print(" Original image counts per artist:")
    print(df['artist'].value_counts())

    #group by artist and sample a fixed number of images per artist
    balanced_df = (
        df.groupby('artist')
          .apply(lambda x: x.sample(n=samples_per_artist, random_state=seed))
          .reset_index(drop=True)
    )

    #save the balanced dataset
    balanced_df.to_csv(output_csv, index=False)
    print(f" Saved balanced dataset to {output_csv}")
    print("Final image counts per artist:")
    print(balanced_df['artist'].value_counts())

if __name__ == "__main__":
    balance_dataset(CSV_PATH, OUTPUT_CSV_PATH, SAMPLES_PER_ARTIST, SEED)

