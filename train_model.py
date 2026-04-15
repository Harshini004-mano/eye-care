"""
Retinal Disease Detection - Model Training Script
Uses ResNet50 with Transfer Learning for 5-class classification
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATASET_DIR   = "dataset"
MODEL_SAVE    = "model/retinal_model.pt"
HISTORY_SAVE  = "model/training_history.json"
PLOT_SAVE     = "static/training_plot.png"
CM_SAVE       = "static/confusion_matrix.png"

NUM_CLASSES   = 5
BATCH_SIZE    = 16
EPOCHS        = 30
LR            = 1e-4
IMG_SIZE      = 224
VAL_SPLIT     = 0.2
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unga folder order-ku yethapadi mathiyachu (Alphabetical Order)
CLASS_NAMES = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "macular_degeneration",
    "normal"
]

CLASS_LABELS = {
    "cataract":             "Cataract",
    "diabetic_retinopathy":   "Diabetic Retinopathy",
    "glaucoma":               "Glaucoma",
    "macular_degeneration":   "Macular Degeneration",
    "normal":                 "Normal"
}

# ─── DATA TRANSFORMS ─────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ─── MODEL ───────────────────────────────────────────────────────────────────
def build_model(num_classes=NUM_CLASSES):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all backbone layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last ResNet block for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model.to(DEVICE)


# ─── TRAINING ────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


# ─── PLOTS ───────────────────────────────────────────────────────────────────
def save_training_plot(history):
    os.makedirs("static", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f1117')

    epochs = range(1, len(history['train_loss']) + 1)
    colors = {'train': '#00d4ff', 'val': '#ff6b6b'}

    for ax in axes:
        ax.set_facecolor('#1a1d2e')
        ax.tick_params(colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    axes[0].plot(epochs, history['train_loss'], color=colors['train'], lw=2, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   color=colors['val'],   lw=2, label='Val Loss', linestyle='--')
    axes[0].set_title('Loss Over Epochs', color='white', fontsize=14, pad=10)
    axes[0].set_xlabel('Epoch', color='#aaaaaa')
    axes[0].set_ylabel('Loss', color='#aaaaaa')
    axes[0].legend(facecolor='#1a1d2e', edgecolor='#333355', labelcolor='white')

    axes[1].plot(epochs, [a*100 for a in history['train_acc']], color=colors['train'], lw=2, label='Train Acc')
    axes[1].plot(epochs, [a*100 for a in history['val_acc']],   color=colors['val'],   lw=2, label='Val Acc', linestyle='--')
    axes[1].set_title('Accuracy Over Epochs', color='white', fontsize=14, pad=10)
    axes[1].set_xlabel('Epoch', color='#aaaaaa')
    axes[1].set_ylabel('Accuracy (%)', color='#aaaaaa')
    axes[1].legend(facecolor='#1a1d2e', edgecolor='#333355', labelcolor='white')

    plt.tight_layout(pad=2)
    plt.savefig(PLOT_SAVE, dpi=120, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  ✓ Training plot saved → {PLOT_SAVE}")


def save_confusion_matrix(all_labels, all_preds):
    os.makedirs("static", exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d2e')

    short = ["Cataract", "DR", "Glaucoma", "Macular", "Normal"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short, yticklabels=short,
                ax=ax, linewidths=0.5, linecolor='#333355')

    ax.set_title('Confusion Matrix', color='white', fontsize=14, pad=12)
    ax.set_xlabel('Predicted', color='#aaaaaa', labelpad=8)
    ax.set_ylabel('Actual',    color='#aaaaaa', labelpad=8)
    ax.tick_params(colors='#cccccc')
    plt.tight_layout()
    plt.savefig(CM_SAVE, dpi=120, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  ✓ Confusion matrix saved → {CM_SAVE}")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print("  Retinal Disease Detection — Model Training")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    os.makedirs("model", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    # Dataset
    full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transforms)
    print(f"  Classes found: {full_dataset.classes}")
    print(f"  Total images : {len(full_dataset)}")

    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Apply val transforms to val split
    val_ds.dataset = datasets.ImageFolder(DATASET_DIR, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Train: {train_size} | Val: {val_size}\n")

    model     = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history    = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val   = 0.0
    best_preds = None
    best_labels= None

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, preds, labels = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        marker = " ★ BEST" if vl_acc > best_val else ""
        print(f"  Epoch [{epoch:02d}/{EPOCHS}]  "
              f"Loss: {tr_loss:.4f}/{vl_loss:.4f}  "
              f"Acc: {tr_acc*100:.1f}%/{vl_acc*100:.1f}%{marker}")

        if vl_acc > best_val:
            best_val    = vl_acc
            best_preds  = preds
            best_labels = labels
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'classes':     full_dataset.classes,
                'val_acc':     best_val,
                'img_size':    IMG_SIZE
            }, MODEL_SAVE)

    print(f"\n  Best Validation Accuracy: {best_val*100:.2f}%")
    print(f"  Model saved → {MODEL_SAVE}\n")

    # Save history
    with open(HISTORY_SAVE, 'w') as f:
        json.dump(history, f, indent=2)

    # Classification report
    class_names_short = [full_dataset.classes[i] for i in range(NUM_CLASSES)]
    print("\n  Classification Report:")
    print(classification_report(best_labels, best_preds, target_names=class_names_short))

    save_training_plot(history)
    save_confusion_matrix(best_labels, best_preds)
    print("\n  Training complete!\n")


if __name__ == "__main__":
    main()
