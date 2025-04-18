import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ArtistDataset(Dataset):
    ''' Dataset to load artist classification data from csv files. each row in the csv file corresponds to an image and its label (artist) '''

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        #create a mapping from artist names to labels
        self.artist_to_label = {artist: label for label, artist in enumerate(self.data['artist'].unique())}
        self.label_to_artist = {label: artist for artist, label in self.artist_to_label.items()}

        #map artist names to labels
        self.data['label'] = self.data['artist'].map(self.artist_to_label)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ''' Get an item from the dataset '''
        row = self.data.iloc[idx]
        img_path = row['file_path']
        label = row['label']

        #load image
        img = Image.open(img_path).convert('RGB')

        #apply transformations if any
        if self.transform:
            img = self.transform(img)

        return img, label