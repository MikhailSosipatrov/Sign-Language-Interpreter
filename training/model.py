"""
Neural network architectures for sign language recognition
"""

import torch
import torch.nn as nn


class LSTMSignLanguageModel(nn.Module):
    """
    Bidirectional LSTM model for sign language gesture recognition

    Args:
        input_size: number of features per frame (225 for MediaPipe)
        hidden_size: LSTM hidden dimension (default: 256)
        num_layers: number of LSTM layers (default: 2)
        num_classes: number of gesture classes (default: 1000)
        dropout: dropout probability (default: 0.3)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 1000,
        dropout: float = 0.3
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: input tensor of shape (batch, sequence_length, input_size)

        Returns:
            logits: (batch, num_classes)
        """
        # Input projection
        x = self.input_proj(x)  # (batch, seq, hidden_size)

        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        # hidden: (num_layers * 2, batch, hidden_size // 2)

        # Take last hidden states from both directions
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        # final_hidden: (batch, hidden_size)

        # Classification
        logits = self.classifier(final_hidden)

        return logits


class TransformerSignLanguageModel(nn.Module):
    """
    Transformer Encoder model for sign language gesture recognition

    Args:
        input_size: number of features per frame (225 for MediaPipe)
        d_model: transformer embedding dimension (default: 256)
        nhead: number of attention heads (default: 8)
        num_layers: number of transformer encoder layers (default: 4)
        dim_feedforward: feedforward network dimension (default: 512)
        num_classes: number of gesture classes (default: 1000)
        dropout: dropout probability (default: 0.2)
        max_seq_length: maximum sequence length (default: 60)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        num_classes: int = 1000,
        dropout: float = 0.2,
        max_seq_length: int = 60
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: input tensor of shape (batch, sequence_length, input_size)

        Returns:
            logits: (batch, num_classes)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)  # (batch, seq, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer encoder
        x = self.transformer(x)  # (batch, seq, d_model)

        # Global mean pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        logits = self.classifier(x)

        return logits


# Test script
if __name__ == "__main__":
    print("Testing models...")

    # Test parameters
    batch_size = 4
    seq_length = 60
    input_size = 225  # MediaPipe keypoints
    num_classes = 1000

    # Create dummy input
    x = torch.randn(batch_size, seq_length, input_size)

    print("\n" + "="*60)
    print("LSTM Model")
    print("="*60)

    lstm_model = LSTMSignLanguageModel(
        input_size=input_size,
        hidden_size=256,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3
    )

    lstm_out = lstm_model(x)
    lstm_params = sum(p.numel() for p in lstm_model.parameters())

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {lstm_out.shape}")
    print(f"Parameters:   {lstm_params:,}")

    print("\n" + "="*60)
    print("Transformer Model")
    print("="*60)

    transformer_model = TransformerSignLanguageModel(
        input_size=input_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=num_classes,
        dropout=0.2,
        max_seq_length=seq_length
    )

    transformer_out = transformer_model(x)
    transformer_params = sum(p.numel() for p in transformer_model.parameters())

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {transformer_out.shape}")
    print(f"Parameters:   {transformer_params:,}")

    print("\n✓ Model test completed!")
