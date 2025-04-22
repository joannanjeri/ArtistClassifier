import pandas as pd

TOP_ARTISTS = 15

#load the train csv file
df = pd.read_csv('data/train.csv')
top_artists = df['artist'].value_counts().head(TOP_ARTISTS).index.tolist()
print(f"Top {TOP_ARTISTS} artists:\n{top_artists}")

#filter the train csv file to only include the top artists
def filter_csv(input_csv, output_csv, top_artists):
    df = pd.read_csv(input_csv)
    df = df[df['artist'].isin(top_artists)]
    df.to_csv(output_csv, index=False)
    print(f"Filtered {input_csv} to {output_csv} ({len(df)} rows)")

#apply the filter to all csv files
for split in ['train', 'val', 'test']:
    filter_csv(f'data/{split}.csv', f'data/{split}_top{TOP_ARTISTS}.csv', top_artists)

#distribution of artists images in the filtered train csv file
filtered_df = pd.read_csv(f'data/train_top{TOP_ARTISTS}.csv')
print("\nImgs per artist in the filtered train set:")
print(filtered_df['artist'].value_counts())



        
