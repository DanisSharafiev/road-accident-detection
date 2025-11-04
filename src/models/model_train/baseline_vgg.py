import torch
import torch.nn as nn
from ...data.data_loader import get_data_loader
from ..model_class.baseline_vgg16  import VGG16Baseline

# Пути к данным
train_paths = {
    "datasets/1/data/train/Accident": 1,
    "datasets/1/data/train/Non Accident": 0
}

val_paths = {
    "datasets/1/data/val/Accident": 1,
    "datasets/1/data/val/Non Accident": 0
}

test_paths = {
    "datasets/1/data/test/Accident": 1,
    "datasets/1/data/test/Non Accident": 0
}


if __name__ == "__main__":
    # Загружаем данные
    train_loader = get_data_loader(image_dict=train_paths, batch_size=16, shuffle=True, num_workers=4)
    val_loader = get_data_loader(image_dict=val_paths, batch_size=16, shuffle=False, num_workers=4)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Инициализация модели
    model = VGG16Baseline(num_classes=2, pretrained=True, freeze_features=True)

    # Настройки обучения
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on device: {device}")

    # Цикл обучения
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
                print(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Training Loss: {epoch_loss:.4f}")

        # ----- Валидация -----
        model.eval()
        val_loss = 0.0
        total = 0
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0.0

        print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}\n")

    # Сохраняем веса
    torch.save(model.state_dict(), "models/vgg16_baseline.pth")
    print("✅ Model saved to models/vgg16_baseline.pth")
