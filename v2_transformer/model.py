import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChessTransformer(nn.Module):
    def __init__(self, vocab_size=16, d_model=256, nhead=8, num_layers=6, max_len=64):
        super(ChessTransformer, self).__init__()
        # vocab_size = 13 (0..12) + margin = 16
        # max_len = 64 squares
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding is crucial for it to know A1 vs H8
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Policy Head now needs to pool the board state?
        # Standard BERT/ViT approach: Use the [CLS] token or Flatten.
        # OR: We can just use the output of the whole sequence flattened?
        # 64 * 256 = 16384 -> 4096 is big but doable.
        # Alternatively, simpler: MaxPool or MeanPool over the 64 squares.
        
        # Let's use Flatten -> Linear for the Policy. It lets it see the whole board relation.
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_len * d_model, 4096)
        )
        
        # Value Head
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_len * d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [batch, 64]
        batch, seq_len = x.size()
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x += self.pos_encoder[:, :seq_len, :]
        
        x = self.transformer_encoder(x)
        
        # x is [batch, 64, d_model]
        # We process the WHOLE board state to decide the move.
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
