import chess
import numpy as np
import torch

# Configuration
# Configuration
board_size_sq = 64
CONTEXT_SIZE = 64 # Fixed for board state (8x8)
# Tokens: 0=Empty, 1-6=White, 7-12=Black, 13=PAD?

def encode_move(move):
    return move.from_square * 64 + move.to_square

def decode_move(index, board):
    from_sq = index // 64
    to_sq = index % 64
    move = chess.Move(from_sq, to_sq)
    # Auto-promote to Queen for simplicity in decoding
    if chess.square_rank(to_sq) in [0, 7]:
        p = board.piece_at(from_sq)
        if p and p.piece_type == chess.PAWN:
            move.promotion = chess.QUEEN
    return move

def board_to_sequence(board):
    # Map pieces to integers
    # Empty=0
    # White: P=1, N=2, B=3, R=4, Q=5, K=6
    # Black: p=7, n=8, b=9, r=10, q=11, k=12
    # We can use piece.piece_type (1..6) and color
    # White: type
    # Black: type + 6
    
    seq = []
    # Always consistent order: A1..H8 (square 0..63)
    for sq in range(64):
        p = board.piece_at(sq)
        if p is None:
            seq.append(0)
        else:
            val = p.piece_type # 1..6
            if p.color == chess.BLACK:
                val += 6
            seq.append(val)
    return seq

def sequence_to_tensor(sequence, device='cuda'):
    # sequence is list of 64 ints
    return torch.tensor([sequence], dtype=torch.long, device=device)

def get_material_score(board):
    # Standard values: P=1, N=3, B=3, R=5, Q=9
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    score = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                score += val
            else:
                score -= val
    return score

