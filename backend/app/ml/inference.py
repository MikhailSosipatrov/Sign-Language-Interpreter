"""
Model inference module for sign language recognition
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add training directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'training'))

from model import LSTMSignLanguageModel, TransformerSignLanguageModel
from .preprocessor import normalize_keypoints, pad_or_trim


class SignLanguagePredictor:
    """
    Class for loading trained model and making predictions

    Args:
        checkpoint_path: path to trained model checkpoint (.pth file)
    """

    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get model arguments
        args = checkpoint['args']
        num_classes = len(checkpoint['idx_to_class'])
        input_size = 225  # MediaPipe keypoints

        # Create model
        if args['model'] == 'lstm':
            self.model = LSTMSignLanguageModel(
                input_size=input_size,
                hidden_size=args['hidden_size'],
                num_layers=args['num_layers'],
                num_classes=num_classes,
                dropout=args['dropout']
            )
        else:  # transformer
            self.model = TransformerSignLanguageModel(
                input_size=input_size,
                d_model=args['hidden_size'],
                nhead=8,
                num_layers=args['num_layers'],
                dim_feedforward=args['hidden_size'] * 2,
                num_classes=num_classes,
                dropout=args['dropout'],
                max_seq_length=args['sequence_length']
            )

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Save metadata
        self.idx_to_class = checkpoint['idx_to_class']
        self.sequence_length = args['sequence_length']
        self.model_type = args['model']

        print(f"✓ Model loaded successfully")
        print(f"  Type: {self.model_type}")
        print(f"  Classes: {num_classes}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def predict(self, keypoints: np.ndarray, top_k: int = 5):
        """
        Make prediction from keypoints

        Args:
            keypoints: numpy array of shape (num_frames, 225)
            top_k: number of top predictions to return

        Returns:
            dictionary with prediction results:
            {
                'predicted_sign': str,
                'confidence': float,
                'top_k': [{'sign': str, 'confidence': float}, ...]
            }
        """
        # Preprocessing
        keypoints = normalize_keypoints(keypoints)
        keypoints = pad_or_trim(keypoints, self.sequence_length)

        # Convert to tensor
        x = torch.FloatTensor(keypoints).unsqueeze(0).to(self.device)
        # x shape: (1, sequence_length, 225)

        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]

        # Format results
        result = {
            'predicted_sign': self.idx_to_class[str(top_indices[0])],
            'confidence': float(top_probs[0]),
            'top_k': [
                {
                    'sign': self.idx_to_class[str(idx)],
                    'confidence': float(prob)
                }
                for idx, prob in zip(top_indices, top_probs)
            ]
        }

        return result


# Test script
if __name__ == "__main__":
    import json

    print("Testing SignLanguagePredictor...")

    # Create dummy keypoints
    dummy_keypoints = np.random.randn(60, 225).astype(np.float32)

    # Test (you need a trained model for this)
    # predictor = SignLanguagePredictor("../../training/models/lstm_baseline/best_model.pth")
    # result = predictor.predict(dummy_keypoints)
    # print(json.dumps(result, indent=2, ensure_ascii=False))

    print("✓ Module test completed")
