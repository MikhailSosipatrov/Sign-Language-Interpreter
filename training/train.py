"""
Training script for sign language recognition models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import argparse
from datetime import datetime

from dataset import create_dataloaders
from model import LSTMSignLanguageModel, TransformerSignLanguageModel


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
):
    """
    Train for one epoch

    Args:
        model: neural network model
        train_loader: training data loader
        criterion: loss function
        optimizer: optimizer
        device: device (cpu or cuda)
        epoch: current epoch number

    Returns:
        average loss, accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    return total_loss / len(train_loader), 100. * correct / total


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
):
    """
    Validate model

    Args:
        model: neural network model
        val_loader: validation data loader
        criterion: loss function
        device: device (cpu or cuda)
        epoch: current epoch number

    Returns:
        average loss, accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc=f"Epoch {epoch} [Val]  "):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(val_loader), 100. * correct / total


def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Create dataloaders
    print("\n" + "="*60)
    print("Creating dataloaders...")
    print("="*60)

    train_loader, test_loader, idx_to_class = create_dataloaders(
        annotations_path=args.annotations,
        keypoints_path=args.keypoints,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_workers=args.num_workers
    )

    num_classes = len(idx_to_class)

    # Infer feature dimension from actual dataset instead of hardcoding.
    try:
        sample_batch, _ = next(iter(train_loader))
    except StopIteration as exc:
        raise RuntimeError("Train dataset is empty after filtering.") from exc
    input_size = int(sample_batch.shape[-1])
    args.input_size = input_size

    # Save args (including inferred input_size)
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nDataset info:")
    print(f"  Classes: {num_classes}")
    print(f"  Input size: {input_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Create model
    print("\n" + "="*60)
    print(f"Creating model: {args.model}")
    print("="*60)

    if args.model == 'lstm':
        model = LSTMSignLanguageModel(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_classes=num_classes,
            dropout=args.dropout
        )
    else:  # transformer
        model = TransformerSignLanguageModel(
            input_size=input_size,
            d_model=args.hidden_size,
            nhead=8,
            num_layers=args.num_layers,
            dim_feedforward=args.hidden_size * 2,
            num_classes=num_classes,
            dropout=args.dropout,
            max_seq_length=args.sequence_length
        )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    best_test_acc = 0
    patience_counter = 0

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    start_time = datetime.now()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        test_loss, test_acc = validate(
            model, test_loader, criterion, device, epoch
        )

        # Scheduler step
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Print metrics
        print(f"\nResults:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        print(f"  LR:    {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'idx_to_class': idx_to_class,
                'args': vars(args)
            }

            torch.save(checkpoint, save_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (test_acc: {test_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    # Training completed
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Duration: {duration}")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Model saved to: {save_dir / 'best_model.pth'}")

    # Save history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"History saved to: {save_dir / 'history.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train sign language recognition model')

    # Data
    parser.add_argument('--annotations', type=str, default='data/raw/annotations.csv',
                        help='path to annotations.csv')
    parser.add_argument('--keypoints', type=str, default='data/raw/slovo_mediapipe.json',
                        help='path to keypoints JSON')
    parser.add_argument('--save_dir', type=str, default='models/lstm_baseline',
                        help='directory to save model')

    # Model
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'transformer'],
                        help='model architecture')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout probability')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--sequence_length', type=int, default=60,
                        help='sequence length (frames)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='early stopping patience')

    # System
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of data loading workers (use 0 for Windows)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    args = parser.parse_args()

    main(args)
