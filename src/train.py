import os 
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader 
from src.dataset import ArtistDataset
from utils.transforms import get_train_transforms, get_val_transforms

#config
TRAIN_CSV = 'data/train.csv'    
VAL_CSV = 'data/val.csv'
BATCH_SIZE = 32
NUM_EPOCHS = 10             #num of epochs to train for
LEARNING_RATE = 1e-4        #rate for the optimizer
NUM_WORKERS = 2             #num of workers for data loading
NUM_CLASSES = 51            #number of artists
MODEL_SAVE_PATH = 'models/resnet50_artist.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(num_classes):
    ''' A pretrained resnet50 model with the final layer replaced for artist classification '''
    model = models.resnet50(pretrained=True)

    #freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False

    #replace the final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    ''' Train the model for num_epochs and validate after each epoch '''
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0 
        total = 0 

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # forward pass for training
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / total
        train_acc = correct / total

        # validate the model
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}]" f" Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | " f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model

def main():
    print("Loading datasets...")
    train_dataset = ArtistDataset(TRAIN_CSV, transform=get_train_transforms())
    val_dataset = ArtistDataset(VAL_CSV, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Building model...")
    model = build_model(NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Training model...")
    trained_model = train(model, train_loader, val_loader, criterion, optimizer, DEVICE, NUM_EPOCHS)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)

    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
    




    