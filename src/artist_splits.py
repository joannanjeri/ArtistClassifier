import os
import random 
import pandas as pd
from sklearn.model_selection import train_test_split

#paths
img_dir = 'data/images'
output_dir = 'data'

VAL_RATIO = 0.15    # 15% of data for validation 
TEST_RATIO = 0.15   # 15% of data for testing
SEED = 50           # random seed for reproducibility 
MAX_IMAGES_PER_ARTIST = None

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def get_image_paths(img_dir, max_imgs_per_artist=None):
    ''' Go through the artist fodler, get image file paths and their corresponding artist names. Returns a dataframe with columns: artist, file_path '''

    dt = []

    #list all artist folders 
    artists = sorted(os.listdir(img_dir))
    for artist in artists:
        artist_path = os.path.join(img_dir, artist)

        #skip if not a directory
        if not os.path.isdir(artist_path):
            continue

        #get all image files in the artist folder
        img_files = [f for f in os.listdir(artist_path) if f.lower().endswith(IMG_EXTENSIONS)]

        #randmomly sample images if max_images_per_artist is set
        if max_imgs_per_artist: 
            random.seed(SEED)
            img_files = random.sample(img_files, min(len(img_files), max_imgs_per_artist))

        #create tuples of (artist, file_path)
        for img_name in img_files:
            full_path = os.path.join(artist_path, img_name)
            dt.append((full_path, artist))

        print(f" Collected {len(img_files)} images for artist: {artist}")

    return pd.DataFrame(dt, columns=['file_path', 'artist'])

def split_and_save(df, output_dir, val_ratio, test_ratio, seed):
    ''' Split the dataframe into train, val, test sets and save them as csv files '''
    
    #split test set 
    train_val_df, test_df = train_test_split(df, test_size=test_ratio, stratify=df['artist'], random_state=seed)

    #split val from the remaining data
    train_df, val_df = train_test_split(train_val_df, test_size=val_ratio/(1-test_ratio), stratify=train_val_df['artist'], random_state=seed)

    #save output dir 
    os.makedirs(output_dir, exist_ok=True)

    #save to csv files
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print("\n Splits saved:")
    print(f" Train: {len(train_df)} images")
    print(f" Val: {len(val_df)} images")
    print(f" Test: {len(test_df)} images")

if __name__ == "__main__":
    print("Collecting image paths...")
    df = get_image_paths(img_dir, max_imgs_per_artist=MAX_IMAGES_PER_ARTIST)

    print("\n Splitting data...")
    split_and_save(df, output_dir, VAL_RATIO, TEST_RATIO, SEED)

    print("\n Done!")