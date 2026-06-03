import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNN(nn.Module):
    def __init__(self, input_size=768, hidden_size=512):
        super(ChessNN, self).__init__()
        # Input: 12 piece types * 64 squares = 768 inputs (Sparse bitboard-like)
        # We can also add extra features like side-to-move, castling rights, etc.
        # For simplicity, let's start with just piece placement + global features?
        # Actually 768 is fine for a simple linear layer start for NNUE-style.
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, 32)
        self.output = nn.Linear(32, 1) # Single scalar evaluation (centipawn)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.output(x)

def fen_to_tensor(fen, device='cpu'):
    """
    Converts a FEN string to a tensor representation.
    Representation: 12 channels (6 white pieces, 6 black pieces) x 64 squares flattened = 768 floats.
    + 1 for side to move? (1=White, -1=Black)
    For this simple version, let's just do 768 board inputs.
    """
    import chess
    board = chess.Board(fen)
    
    # 64 squares, 12 piece types
    # P N B R Q K p n b r q k
    piece_map = board.piece_map()
    
    # Map piece symbols to index 0-11
    # P=1, N=2, ..., k=12. 
    # Let's map White P (1) -> 0, N(2)->1... K(6)->5
    # Black p(1) -> 6 ... k(6)->11
    
    # Tensor: [768]
    x = torch.zeros(768, dtype=torch.float32, device=device)
    
    for square, piece in piece_map.items():
        # piece.color: True for White, False for Black
        # piece.piece_type: 1(P) to 6(K)
        
        offset = 0 if piece.color else 6
        idx = (piece.piece_type - 1) + offset # 0-11
        
        # position in flat 768: idx * 64 + square
        # strictly speaking NNUE does "King-Relative" but detailed simple input is:
        # 12 planes of 64 squares flattened.
        
        flat_idx = idx * 64 + square
        x[flat_idx] = 1.0
        
    return x
