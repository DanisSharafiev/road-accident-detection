import torch
import torch.nn as nn
from ...data.data_loader import get_data_loader
from ..model_class.baseline_cnn import BaselineCNN
import os

train_paths = {
    "datasets/train/Accident" : 1,
    "datasets/train/Non Accident" : 0
}

val_paths = {
    "datasets/val/Accident" : 1,
    "datasets/val/Non Accident" : 0
}

test_paths = {
    "datasets/test/Accident" : 1,
    "datasets/test/Non Accident" : 0
}


if __name__ == "__main__":
    dataloader = get_data_loader(image_dict=train_paths, batch_size=32, shuffle=True, num_workers=4, augment=True)
    val_dataloader = get_data_loader(image_dict=val_paths, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader = get_data_loader(image_dict=test_paths, batch_size=32, shuffle=False, num_workers=4)
    print(f"Number of batches: {len(dataloader)}")

    model = BaselineCNN(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = model.to(device)
    print(f"Training on device: {device}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        validation_loss = 0.0
        total = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        validation_loss /= len(val_dataloader.dataset)
        validation_accuracy = correct / total if total > 0 else 0
        print(f"Validation Loss: {validation_loss:.4f}, Accuracy: {validation_accuracy:.4f}")

    # test the model
    total = 0
    correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            total += labels.size(0)
            TP += ((torch.max(outputs, 1)[1] == 1) & (labels == 1)).sum().item()
            TN += ((torch.max(outputs, 1)[1] == 0) & (labels == 0)).sum().item()
            FP += ((torch.max(outputs, 1)[1] == 1) & (labels == 0)).sum().item()
            FN += ((torch.max(outputs, 1)[1] == 0) & (labels == 1)).sum().item()

    print(f"Test TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")    
    print(f"Accuracy: {correct/total:.4f}")

    os.makedirs("dict_models", exist_ok=True)

    torch.save(model.state_dict(), "dict_models/baseline_cnn.pth")