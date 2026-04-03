# train_eeg.py
import torch
import torch.nn as nn

from dataloader import load_deap_eeg_loaders
from model import EEGEmotionNet
import random
import numpy as np

DATA_PATH = r"C:\chou\Deap_eeg\deap_eeg_segments_baseline.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X, y in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += X.size(0)

    return total_loss / total_samples, total_correct / total_samples


def eval_model(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += X.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():

    train_loader, test_loader = load_deap_eeg_loaders(
        npz_path=DATA_PATH,
        test_size=0.3,        # 7:3 hold-out
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    model = EEGEmotionNet(n_channels=32, segment_len=128, n_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_test_acc = 0.0
    save_path = r"C:\chou\Deap_eeg\eeg_localglobal_best.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = eval_model(model, test_loader, criterion)

        print(
            f"[Epoch {epoch:03d}/{EPOCHS}] "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Test: loss={test_loss:.4f} acc={test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved (acc={best_test_acc:.4f})")

    print(f"\nBest test acc: {best_test_acc:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
