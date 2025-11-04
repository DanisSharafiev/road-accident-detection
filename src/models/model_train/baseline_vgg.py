import torch
import torch.nn as nn
from ...data.data_loader import get_data_loader
from ..model_class.baseline_vgg16  import VGG16Baseline
import torchvision.transforms as transforms
import os

train_paths = {
    "datasets/train/Accident": 1,
    "datasets/train/Non Accident": 0
}

val_paths = {
    "datasets/val/Accident": 1,
    "datasets/val/Non Accident": 0
}

test_paths = {
    "datasets/test/Accident": 1,
    "datasets/test/Non Accident": 0
}



if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    random_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)], p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    train_loader = get_data_loader(image_dict=train_paths, batch_size=32, shuffle=True, num_workers=4, augment=random_transform)
    val_loader = get_data_loader(image_dict=val_paths, batch_size=32, shuffle=False, num_workers=4, augment=transform)
    test_loader = get_data_loader(image_dict=test_paths, batch_size=32, shuffle=False, num_workers=4, augment=transform)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = VGG16Baseline(num_classes=2, pretrained=True, freeze_features=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)
    best_model = None
    best_val_loss = float('inf')
    print(f"Training on device: {device}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} | Train Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Training Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        total = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if batch_idx % 10 == 0:
                    print(f"[Epoch {epoch+1}] Batch {batch_idx}/{len(val_loader)} | Val Loss: {loss.item():.4f}")

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0.0

        print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            print("Best model updated.")

    # testing
    total = 0
    correct = 0
    if best_model is not None:
        model.load_state_dict(best_model)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Test accuracy: {:.4f}".format(correct / total if total > 0 else 0.0))

    os.makedirs("dict_models", exist_ok=True)

    torch.save(best_model, "dict_models/vgg16_baseline.pth")
    print("Model saved to dict_models/vgg16_baseline.pth")
