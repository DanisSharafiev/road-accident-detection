import torch
import torch.nn as nn
from ...data.data_loader import get_data_loader
from ..model_class.baseline_vgg16 import VGG16Baseline


train_paths = {
    "datasets/1/data/train/Accident":      1,
    "datasets/1/data/train/Non Accident":  0
}

val_paths = {
    "datasets/1/data/val/Accident":        1,
    "datasets/1/data/val/Non Accident":    0
}

test_paths = {
    "datasets/1/data/test/Accident":       1,
    "datasets/1/data/test/Non Accident":   0
}


def calculate_accuracy(outputs, labels):
    """Возвращает (correct, total) для вычисления accuracy."""
    _, preds = torch.max(outputs.data, 1)
    total   = labels.size(0)
    correct = (preds == labels).sum().item()
    return correct, total


def print_epoch_stats(ep, num_ep, tr_loss, tr_acc, val_loss, val_acc):
    """вывод статистики эпохи."""
    bar = "=" * 80
    print(bar)
    print(f"EPOCH [{ep+1:2d}/{num_ep:2d}]")
    print(f"   Train      Loss: {tr_loss:.6f} | Acc: {tr_acc:.4f} ({tr_acc*100:5.2f}%)")
    print(f"   Validation Loss: {val_loss:.6f} | Acc: {val_acc:.4f} ({val_acc*100:5.2f}%)")
    print(bar)


# ──────────────────────── Точка входа ────────────────────────
if __name__ == "__main__":
    # Даталоадеры (num_workers=0 to avoid multiprocessing issues on macOS)
    train_loader = get_data_loader(train_paths, batch_size=16, shuffle=True,  num_workers=0)
    val_loader   = get_data_loader(val_paths,   batch_size=16, shuffle=False, num_workers=0)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Модель
    model = VGG16Baseline(num_classes=2, pretrained=True, freeze_features=True)

    # Обучение
    criterion   = nn.CrossEntropyLoss()
    optimizer   = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs  = 1
    # VGG16 has issues with MPS adaptive pooling, so we use CPU on Apple Silicon
    # For NVIDIA GPU, CUDA will be used
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    model       = model.to(device)

    print(f"Device: {device}")
    print(f"Params total/trainable: "
          f"{sum(p.numel() for p in model.parameters())}/"
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # История
    train_losses, val_losses = [], []
    train_accs,  val_accs   = [], []

    print("\nSTART TRAINING"); print("=" * 80)

    for epoch in range(num_epochs):
        # ---------- Train ----------
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for b, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            optimizer.zero_grad()
            outs  = model(imgs)
            loss  = criterion(outs, lbls)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * imgs.size(0)
            c, t     = calculate_accuracy(outs, lbls)
            correct += c; total += t

            if b % 10 == 0:
                batch_acc = c / t
                print(f"[Ep {epoch+1}/{num_epochs}] Batch {b:3d}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f} | Acc: {batch_acc:.4f}")

        epoch_tr_loss = run_loss / len(train_loader.dataset)
        epoch_tr_acc  = correct / total

        # ---------- Validation ----------
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outs = model(imgs)
                loss = criterion(outs, lbls)
                v_loss += loss.item() * imgs.size(0)
                c, t   = calculate_accuracy(outs, lbls)
                v_correct += c; v_total += t

        epoch_val_loss = v_loss / len(val_loader.dataset)
        epoch_val_acc  = v_correct / v_total

        # Сохраняем историю и выводим статистику
        train_losses.append(epoch_tr_loss); train_accs.append(epoch_tr_acc)
        val_losses.append(epoch_val_loss);   val_accs.append(epoch_val_acc)
        print_epoch_stats(epoch, num_epochs, epoch_tr_loss, epoch_tr_acc,
                          epoch_val_loss, epoch_val_acc)

    import os

    # ---------- Итог ----------
    best_acc  = max(val_accs)
    best_ep   = val_accs.index(best_acc) + 1
    print(f"\nBest val-accuracy: {best_acc:.4f} at epoch {best_ep}")

    # Создаем директорию, если её нет
    os.makedirs("models", exist_ok=True)

    torch.save({
        "epoch":          num_epochs,
        "model_state":    model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "train_losses":   train_losses,
        "train_accs":     train_accs,
        "val_losses":     val_losses,
        "val_accs":       val_accs,
        "best_acc":       best_acc,
        "best_epoch":     best_ep
    }, "models/vgg16_baseline.pth")

    print("Saved to  models/vgg16_baseline.pth")
