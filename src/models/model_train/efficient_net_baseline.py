import torch
import torch.nn as nn
from ...data.data_loader import get_data_loader
from ..model_class.efficient_net_baseline import EfficientNetBaseline

from sklearn.metrics import precision_score, recall_score, confusion_matrix

# ──────────────────────── Paths ────────────────────────
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

# ────────────────── Helper functions ──────────────────
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs.data, 1)
    total   = labels.size(0)
    correct = (preds == labels).sum().item()
    return correct, total


def print_epoch_stats(ep, num_ep, tr_loss, tr_acc, val_loss, val_acc, val_prec, val_rec, conf_mat):
    bar = "=" * 80
    print(bar)
    print(f"📊  EPOCH [{ep+1:2d}/{num_ep:2d}]")
    print(f"   🔹 Train      Loss: {tr_loss:.6f} | Acc: {tr_acc:.4f} ({tr_acc*100:5.2f}%)")
    print(f"   🔹 Validation Loss: {val_loss:.6f} | Acc: {val_acc:.4f} ({val_acc*100:5.2f}%)")
    print(f"   🔹 Validation Precision: {val_prec}")
    print(f"   🔹 Validation Recall:    {val_rec}")
    print(f"   🔹 Confusion Matrix:\n{conf_mat}")
    print(bar)

# ──────────────────────── Entry point ────────────────────────
if __name__ == "__main__":
    train_loader = get_data_loader(train_paths, batch_size=16, shuffle=True,  num_workers=4)
    val_loader   = get_data_loader(val_paths,   batch_size=16, shuffle=False, num_workers=4)
    print(f"📁 Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = EfficientNetBaseline(
        num_classes=2,
        pretrained=True,
        freeze_features=True,
        model_variant='b0'  # or 'b1', 'b2' for stronger models
    )

    criterion   = nn.CrossEntropyLoss()
    optimizer   = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs  = 10
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model       = model.to(device)

    print(f"🚀 Device: {device}")
    print(f"🎯 Params total/trainable: "
          f"{sum(p.numel() for p in model.parameters())}/"
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_losses, val_losses = [], []
    train_accs,  val_accs   = [], []

    print("\n🔥 START TRAINING")
    print("=" * 80)

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

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outs = model(imgs)
                loss = criterion(outs, lbls)
                v_loss += loss.item() * imgs.size(0)
                c, t   = calculate_accuracy(outs, lbls)
                v_correct += c; v_total += t

                preds = torch.argmax(outs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbls.cpu().numpy())

        epoch_val_loss = v_loss / len(val_loader.dataset)
        epoch_val_acc  = v_correct / v_total

        val_precision = precision_score(all_labels, all_preds, average=None)
        val_recall = recall_score(all_labels, all_preds, average=None)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        train_losses.append(epoch_tr_loss)
        train_accs.append(epoch_tr_acc)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print_epoch_stats(
            epoch, num_epochs,
            epoch_tr_loss, epoch_tr_acc,
            epoch_val_loss, epoch_val_acc,
            val_precision, val_recall,
            conf_matrix
        )

    best_acc = max(val_accs)
    best_ep = val_accs.index(best_acc) + 1
    print(f"\n🏆 Best val-accuracy: {best_acc:.4f} at epoch {best_ep}")

    import os
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
    }, "models/efficientnet_baseline.pth")

    print("✅  Saved to  models/efficientnet_baseline.pth")
