import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.dataset import ArtistDataset
from utils.transforms import get_val_transforms
from sklearn.utils.multiclass import unique_labels

#config
TEST_CSV = 'data/test.csv'
MODEL_SAVE_PATH = 'models/resnet50_artist.pth'
BATCH_SIZE = 32
NUM_WORKERS = 2
NUM_CLASSES = 49
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(num_classes):
    ''' A pretrained resnet50 model with the final layer replaced for artist classification '''

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    #replace the final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def evaluate(model, data_loader, device, label_names):
    ''' Evaluate the model on the test set and print classification report '''

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    #overall accuracy
    correct = sum(p ==t for p, t in zip(all_preds, all_labels))
    total = len(all_labels)
    accuracy = correct / total
    print(f"\n Test Accuracy: {accuracy * 100:.2f}%")

    #classification report
    print("\n Classification Report:")
    # labels = sorted(list(unique_labels(all_labels, all_preds)))
    # print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))

    labels = sorted(list(unique_labels(all_labels, all_preds)))

    if len(labels) != len(label_names):
        print(f" Detected {len(label_names)} label names.")
        print(f" Switching to int class indices for classification report...")
        print(classification_report(all_labels, all_preds, zero_division=0))
    else:
        print(classification_report(all_labels, all_preds, labels=labels, target_names=label_names, zero_division=0))

    #confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('report/50classes_confusion_matrix.png', dpi=300)
    print("Confusion matrix saved")

def main():
    print("Loading test dataset...")
    test_dataset = ArtistDataset(TEST_CSV, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("Loading model...")
    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    #pass label names to evaluate function
    # label_names = list(test_dataset.artist_to_label.keys())
    label_names = [artist for artist, _ in sorted(test_dataset.artist_to_label.items(), key=lambda x: x[1])]

    print("Evaluating model...")
    evaluate(model, test_loader, DEVICE, label_names)

if __name__ == "__main__":
    main()
    print("Running evaluation script...")
    print("Evaluation complete!")