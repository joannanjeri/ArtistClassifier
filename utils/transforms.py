from torchvision import transforms

#imagenet mean and std values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(img_size=224):
    ''' Transforms for training dataset
    - Resize to img_size
    - Random horizontal flip
    - Convert to tensor
    - Normalize with imagenet mean and std
    '''

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_val_transforms(img_size=224):
    ''' Transforms for validation and testing dataset'''
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])  