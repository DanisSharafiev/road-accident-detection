import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    """Custom dataset for loading images and labels from a dictionary of image paths and labels."""
    def __init__(self, 
                 image_dict : dict[str, int], 
                 transform : transforms.Compose | None = None
                 ) -> None:
        self.image_dict = image_dict
        self.transform = transform
        self.image_files = []
        self.labels = []
        for path, label in image_dict.items():
            for img_file in os.listdir(path):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    self.image_files.append(os.path.join(path, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB') 
        
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

def get_data_loader(image_dict : dict[str, int], 
                    batch_size : int = 32, 
                    shuffle : bool = True, 
                    num_workers : int = 4,
                    augment: transforms.Compose | bool = False
                    ) -> DataLoader:
    """returns a pytorch DataLoader for the given image dictionary.\n
    image_dict: dictionary with image paths as keys and labels as values"""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    if augment is True:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(512, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)], p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
    
    if augment and isinstance(augment, transforms.Compose):
        transform = augment

    dataset = CustomDataset(image_dict=image_dict, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return data_loader
