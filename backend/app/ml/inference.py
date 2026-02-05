"""
Model inference module for sign language recognition.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add training directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "training"))

from model import LSTMSignLanguageModel, TransformerSignLanguageModel
from .preprocessor import normalize_keypoints, pad_or_trim


class SignLanguagePredictor:
    """
    Class for loading trained model and making predictions.

    Args:
        checkpoint_path: path to trained model checkpoint (.pth file)
    """

    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        args = checkpoint["args"]
        num_classes = len(checkpoint["idx_to_class"])
        self.input_size = int(args.get("input_size", 189))

        if args["model"] == "lstm":
            self.model = LSTMSignLanguageModel(
                input_size=self.input_size,
                hidden_size=args["hidden_size"],
                num_layers=args["num_layers"],
                num_classes=num_classes,
                dropout=args["dropout"],
            )
        else:
            self.model = TransformerSignLanguageModel(
                input_size=self.input_size,
                d_model=args["hidden_size"],
                nhead=8,
                num_layers=args["num_layers"],
                dim_feedforward=args["hidden_size"] * 2,
                num_classes=num_classes,
                dropout=args["dropout"],
                max_seq_length=args["sequence_length"],
            )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.idx_to_class = checkpoint["idx_to_class"]
        self.sequence_length = int(args["sequence_length"])
        self.model_type = args["model"]

        print("Model loaded successfully")
        print(f"  Type: {self.model_type}")
        print(f"  Classes: {num_classes}")
        print(f"  Input size: {self.input_size}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _class_name(self, idx: int) -> str:
        # Checkpoint mapping can have int or string keys.
        if idx in self.idx_to_class:
            return self.idx_to_class[idx]
        return self.idx_to_class.get(str(idx), str(idx))

    def predict(self, keypoints: np.ndarray, top_k: int = 5):
        """
        Make prediction from keypoints.

        Args:
            keypoints: numpy array of shape (num_frames, input_size)
            top_k: number of top predictions to return

        Returns:
            {
                "predicted_sign": str,
                "confidence": float,
                "top_k": [{"sign": str, "confidence": float}, ...]
            }
        """
        if keypoints.ndim != 2:
            raise ValueError(f"Expected 2D keypoints array, got shape {keypoints.shape}")
        if keypoints.shape[1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, got {keypoints.shape[1]}"
            )

        keypoints = normalize_keypoints(keypoints)
        keypoints = pad_or_trim(keypoints, self.sequence_length)

        x = torch.FloatTensor(keypoints).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            k = min(top_k, probs.shape[1])
            top_probs, top_indices = torch.topk(probs, k=k, dim=1)
            top_probs = top_probs.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]

        return {
            "predicted_sign": self._class_name(int(top_indices[0])),
            "confidence": float(top_probs[0]),
            "top_k": [
                {"sign": self._class_name(int(idx)), "confidence": float(prob)}
                for idx, prob in zip(top_indices, top_probs)
            ],
        }


if __name__ == "__main__":
    print("Testing SignLanguagePredictor module...")
    dummy_keypoints = np.random.randn(60, 189).astype(np.float32)
    print(f"Dummy input shape: {dummy_keypoints.shape}")
    print("Module test completed")
