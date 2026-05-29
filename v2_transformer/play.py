import pygame
import chess
import torch
import os
import sys
import numpy as np
from model import ChessTransformer
from mcts import MCTS
from utils import encode_move

# Constants
WIDTH, HEIGHT = 600, 600
SQ_SIZE = WIDTH // 8
FPS = 30
ASSETS_DIR = "../assets"

# Colors
WHITE_COL = (240, 217, 181)
BLACK_COL = (181, 136, 99)
HIGHLIGHT = (100, 255, 100, 100)
SELECTED_COL = (0, 255, 0, 100)
HIGHLIGHT_LAST_MOVE_COL = (255, 255, 0)

# Global Images
IMAGES = {}

def load_images():
    global IMAGES
    pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for p in pieces:
        path = os.path.join(ASSETS_DIR, f"{p}.png")
        if os.path.exists(path):
            img = pygame.image.load(path)
            IMAGES[p] = pygame.transform.smoothscale(img, (SQ_SIZE, SQ_SIZE))
        else:
            print(f"Warning: Missing asset {path}")

def draw_board(screen, board, selected_sq=None, last_move=None):
    # Draw Squares
    for r in range(8):
        for c in range(8):
            color = WHITE_COL if (r + c) % 2 == 0 else BLACK_COL
            pygame.draw.rect(screen, color, (c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
            
            # Highlight Last Move
            if last_move:
                 if last_move.from_square == chess.square(c, 7-r) or last_move.to_square == chess.square(c, 7-r):
                     s = pygame.Surface((SQ_SIZE, SQ_SIZE))
                     s.set_alpha(100)
                     s.fill(HIGHLIGHT_LAST_MOVE_COL)
                     screen.blit(s, (c*SQ_SIZE, r*SQ_SIZE))

            # Highlight Selection
            if selected_sq is not None and selected_sq == chess.square(c, 7-r):
                s = pygame.Surface((SQ_SIZE, SQ_SIZE))
                s.set_alpha(150)
                s.fill(SELECTED_COL)
                screen.blit(s, (c*SQ_SIZE, r*SQ_SIZE))

    # Draw Pieces
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            color_prefix = 'w' if piece.color == chess.WHITE else 'b'
            symbol_map = {'P':'P', 'N':'N', 'B':'B', 'R':'R', 'Q':'Q', 'K':'K'}
            key = f"{color_prefix}{symbol_map[piece.symbol().upper()]}"
            
            if key in IMAGES:
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                x = file * SQ_SIZE
                y = (7 - rank) * SQ_SIZE
                screen.blit(IMAGES[key], (x, y))

def get_square_under_mouse(pos):
    x, y = pos
    col = x // SQ_SIZE
    row = y // SQ_SIZE
    # Row 0 is Rank 8? No, Row 0 in pygame is Y=0 (Top).
    # Board Rank 7 is top.
    rank = 7 - row
    file = col
    if 0 <= file <= 7 and 0 <= rank <= 7:
        return chess.square(file, rank)
    return None

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Play vs ChessGPT")
    clock = pygame.time.Clock()
    
    load_images()
    
    # Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on {device}...")
    model = ChessTransformer().to(device)
    if os.path.exists("transformer_model.pth"):
        model.load_state_dict(torch.load("transformer_model.pth", map_location=device, weights_only=True))
        print("Model loaded.")
    else:
        print("Warning: No model found! AI will play randomly.")
    model.eval()
    
    board = chess.Board()
    selected_sq = None
    player_color = chess.WHITE # User is White
    
    running = True
    game_over = False
    
    while running:
        turn = board.turn
        
        # Event Handling
        if not game_over and turn == player_color:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    sq = get_square_under_mouse(e.pos)
                    if sq is not None:
                        if selected_sq is None:
                            # Select piece
                            p = board.piece_at(sq)
                            if p and p.color == player_color:
                                selected_sq = sq
                        else:
                            # Move attempt
                            move = chess.Move(selected_sq, sq)
                            # Check promotion (auto-queen for simplicity)
                            if board.piece_at(selected_sq).piece_type == chess.PAWN:
                                if (player_color == chess.WHITE and chess.square_rank(sq) == 7) or \
                                   (player_color == chess.BLACK and chess.square_rank(sq) == 0):
                                   move = chess.Move(selected_sq, sq, promotion=chess.QUEEN)
                                   
                            if move in board.legal_moves:
                                board.push(move)
                                selected_sq = None
                            else:
                                # Deselect or reselect
                                p = board.piece_at(sq)
                                if p and p.color == player_color:
                                    selected_sq = sq
                                else:
                                    selected_sq = None
                                    
        else:
            # AI Turn or Game Over (But keep loop running for Quit)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
            
            if not game_over and turn != player_color:
                # Draw before thinking
                last_move = board.peek() if board.move_stack else None
                draw_board(screen, board, selected_sq, last_move)
                pygame.display.flip()
                
                print("AI Thinking...")
                # MCTS automatically uses board_to_sequence via internal import, 
                # but if we used raw model inference we'd need it.
                # Since play.py uses MCTS class, and MCTS class was already updated to use board_to_sequence,
                # we just need to make sure play.py imports aren't broken.
                
                mcts = MCTS(model, device)
                root = mcts.search(board, simulations=100) # Fast play
                moves, probs = mcts.get_action_probs(root, temperature=0.1)
                
                # Pick move
                if not moves:
                     print("AI Resigns (No moves)")
                     game_over = True
                else:
                    choice_idx = np.random.choice(len(moves), p=probs)
                    best_move_idx = moves[choice_idx]
                    
                    found = None
                    for m in board.legal_moves:
                        if encode_move(m) == best_move_idx:
                            found = m
                            break
                    if found:
                        board.push(found)
                    else:
                        print("AI Error: Generated illegal move index.")
                        # Fallback random
                        import random
                        board.push(random.choice(list(board.legal_moves)))
                        
        if board.is_game_over():
            game_over = True
            pygame.display.set_caption(f"Game Over: {board.result()}")
        
        # Draw Loop
        last_move = board.peek() if board.move_stack else None
        draw_board(screen, board, selected_sq, last_move)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
