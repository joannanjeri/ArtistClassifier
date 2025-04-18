from src.dataset import ArtistDataset
from utils.transforms import get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader

#load the dataset
train_dataset = ArtistDataset(csv_file='data/train.csv', transform=get_train_transforms())

#create a dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

imgs, labels = next(iter(train_loader))
print(imgs.shape)
print(labels[:5])

