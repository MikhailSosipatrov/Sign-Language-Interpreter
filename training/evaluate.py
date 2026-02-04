"""
Evaluation script for trained sign language recognition models
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm

from dataset import create_dataloaders
from model import LSTMSignLanguageModel, TransformerSignLanguageModel


def evaluate_model(model, test_loader, device, idx_to_class):
    """
    Evaluate model on test set

    Args:
        model: trained model
        test_loader: test data loader
        device: device (cpu or cuda)
        idx_to_class: mapping from index to class name

    Returns:
        dictionary with evaluation results
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("Evaluating model...")
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Top-1 accuracy
    accuracy = (all_preds == all_labels).mean() * 100

    # Top-5 accuracy
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_correct = np.array([
        label in top5
        for label, top5 in zip(all_labels, top5_preds)
    ])
    top5_accuracy = top5_correct.mean() * 100

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {len(all_labels)}")
    print(f"Top-1 Accuracy: {accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def plot_confusion_matrix(cm, save_path=None, top_k=50):
    """
    Plot confusion matrix for top-K most frequent classes

    Args:
        cm: confusion matrix
        save_path: path to save figure
        top_k: number of top classes to display
    """
    # Select top-K most frequent classes
    top_indices = cm.sum(axis=1).argsort()[-top_k:][::-1]
    cm_top = cm[top_indices][:, top_indices]

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_top, cmap='Blues', fmt='d', cbar=True)
    plt.title(f'Confusion Matrix (Top-{top_k} classes)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix saved to: {save_path}")

    plt.close()


def find_worst_classes(cm, idx_to_class, top_k=20):
    """
    Find classes with worst accuracy

    Args:
        cm: confusion matrix
        idx_to_class: mapping from index to class name
        top_k: number of worst classes to show
    """
    class_accuracies = []

    for i in range(len(cm)):
        total = cm[i].sum()
        correct = cm[i, i]
        acc = correct / total if total > 0 else 0
        class_name = idx_to_class.get(i, f"class_{i}")
        class_accuracies.append((i, acc, class_name))

    # Sort by accuracy (ascending)
    class_accuracies.sort(key=lambda x: x[1])

    print(f"\n{top_k} WORST PERFORMING CLASSES:")
    print("-" * 60)
    for i, (class_idx, acc, class_name) in enumerate(class_accuracies[:top_k]):
        print(f"{i+1:2d}. {class_name:30s} {acc*100:6.1f}%")


def find_best_classes(cm, idx_to_class, top_k=20):
    """
    Find classes with best accuracy

    Args:
        cm: confusion matrix
        idx_to_class: mapping from index to class name
        top_k: number of best classes to show
    """
    class_accuracies = []

    for i in range(len(cm)):
        total = cm[i].sum()
        correct = cm[i, i]
        acc = correct / total if total > 0 else 0
        class_name = idx_to_class.get(i, f"class_{i}")
        class_accuracies.append((i, acc, class_name))

    # Sort by accuracy (descending)
    class_accuracies.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{top_k} BEST PERFORMING CLASSES:")
    print("-" * 60)
    for i, (class_idx, acc, class_name) in enumerate(class_accuracies[:top_k]):
        print(f"{i+1:2d}. {class_name:30s} {acc*100:6.1f}%")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load data
    print("\nLoading data...")
    _, test_loader, idx_to_class = create_dataloaders(
        annotations_path=args.annotations,
        keypoints_path=args.keypoints,
        batch_size=args.batch_size,
        sequence_length=checkpoint['args']['sequence_length'],
        num_workers=args.num_workers
    )

    # Create model
    num_classes = len(idx_to_class)
    input_size = int(checkpoint['args'].get('input_size', 225))

    print(f"\nCreating model: {checkpoint['args']['model']}")

    if checkpoint['args']['model'] == 'lstm':
        model = LSTMSignLanguageModel(
            input_size=input_size,
            hidden_size=checkpoint['args']['hidden_size'],
            num_layers=checkpoint['args']['num_layers'],
            num_classes=num_classes,
            dropout=checkpoint['args']['dropout']
        )
    else:  # transformer
        model = TransformerSignLanguageModel(
            input_size=input_size,
            d_model=checkpoint['args']['hidden_size'],
            num_layers=checkpoint['args']['num_layers'],
            num_classes=num_classes,
            dropout=checkpoint['args']['dropout'],
            max_seq_length=checkpoint['args']['sequence_length']
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Evaluate
    results = evaluate_model(model, test_loader, device, idx_to_class)

    # Save directory
    save_dir = Path(args.checkpoint).parent

    # Plot confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'],
        save_path=save_dir / 'confusion_matrix.png',
        top_k=args.top_k_cm
    )

    # Find worst and best classes
    find_worst_classes(results['confusion_matrix'], idx_to_class, top_k=20)
    find_best_classes(results['confusion_matrix'], idx_to_class, top_k=20)

    # Save results
    results_path = save_dir / 'evaluation_results.npy'
    np.save(results_path, results)
    print(f"\n✓ Results saved to: {results_path}")

    # Save metrics to text file
    metrics_path = save_dir / 'evaluation_metrics.txt'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {checkpoint['args']['model']}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Total samples: {len(results['labels'])}\n")
        f.write(f"Top-1 Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%\n")

    print(f"✓ Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to model checkpoint')
    parser.add_argument('--annotations', type=str, default='data/raw/annotations.csv',
                        help='path to annotations.csv')
    parser.add_argument('--keypoints', type=str, default='data/raw/slovo_mediapipe.json',
                        help='path to keypoints JSON')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (use 0 for Windows)')
    parser.add_argument('--top_k_cm', type=int, default=50,
                        help='number of classes to show in confusion matrix')

    args = parser.parse_args()
    main(args)
