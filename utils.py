import chess
import numpy as np
import torch

# 8x8x12 representation
# Channels 0-5: P, N, B, R, Q, K (White)
# Channels 6-11: P, N, B, R, Q, K (Black)
# This is a simplified input (AlphaZero uses historical planes + special planes)
def board_to_tensor(board):
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Map piece types to channel offsets
    # White pieces: 1-6 -> 0-5
    # Black pieces: 1-6 -> 6-11
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Row and Col calculation
            # chess.A1 = 0, chess.H8 = 63
            # In numpy, we want [row, col]. 
            # Rank 1 is usually bottom, but for CNN we map naturally.
            # let's map rank 0->7, file 0->7
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            
            piece_type = piece.piece_type # 1=P, 2=N, ... 6=K
            color_offset = 0 if piece.color == chess.WHITE else 6
            
            # channel index
            channel = (piece_type - 1) + color_offset
            planes[channel, rank, file] = 1.0
            
    return planes

def encode_move(move):
    # Simplified encoding: 64 * 64 = 4096 possibilities (from_square -> to_square)
    # This ignores promotion type (always queens or context dependent)
    return move.from_square * 64 + move.to_square

def decode_move(index, board):
    # Decode logic
    from_sq = index // 64
    to_sq = index % 64
    
    # Attempt to create a legal move
    # We must handle promotions. If the move is a promotion, we try to create a Queen promotion.
    move = chess.Move(from_sq, to_sq)
    
    # Check if this strictly corresponds to a promotion
    if chess.square_rank(to_sq) in [0, 7]:
        # Possible promotion
        p = board.piece_at(from_sq)
        if p and p.piece_type == chess.PAWN:
            move.promotion = chess.QUEEN # simple assumption
            
    return move
