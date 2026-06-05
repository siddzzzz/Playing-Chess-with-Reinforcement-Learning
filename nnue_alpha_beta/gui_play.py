import pygame
import chess
import os
import sys
import time
from engine import NNUEEngine
import argparse

# Constants
WIDTH, HEIGHT = 800, 800
SQ_SIZE = WIDTH // 8
ASSET_DIR = "../assets"
FPS = 60

# Colors
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
HIGHLIGHT = (255, 255, 0, 100) # Yellow transparent
MOVE_HINT = (0, 255, 0, 100)

def load_assets():
    # Background is now generated procedurally
    bg = None

    pieces = {}
    piece_map = {
        'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
        'p': 'P', 'n': 'N', 'b': 'B', 'r': 'R', 'q': 'Q', 'k': 'K'
    }
    
    for symbol in piece_map.keys():
        color = 'w' if symbol.isupper() else 'b'
        filename = f"{color}{piece_map[symbol]}.png"
        path = os.path.join(ASSET_DIR, filename)
        if os.path.exists(path):
            try:
                img = pygame.image.load(path)
                img = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
                pieces[symbol] = img
            except pygame.error as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Warning: {filename} not found.")
            
    return bg, pieces

def draw_board(screen, bg=None):
    # Modern Blue/White Theme
    colors = [pygame.Color("#EAE9D2"), pygame.Color("#4B7399")]
    
    for r in range(8):
        for c in range(8):
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color, (c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))
            
    # Optional: Draw coordinate labels could go here, but keeping it simple for now.

def draw_pieces(screen, board, pieces):
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            symbol = piece.symbol() # 'P', 'p' etc.
            if symbol in pieces:
                # Rank 0 (1) is bottom, Rank 7 (8) is top
                # Screen Y: Top (0) to Bottom (Height)
                # Rank 7 -> y=0. Rank 0 -> y=7*SQ
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                
                x = file * SQ_SIZE
                y = (7 - rank) * SQ_SIZE
                
                screen.blit(pieces[symbol], (x, y))

def get_square_under_mouse(pos):
    x, y = pos
    col = x // SQ_SIZE
    row = y // SQ_SIZE
    file = col
    rank = 7 - row
    if 0 <= file <= 7 and 0 <= rank <= 7:
        return chess.square(file, rank)
    return None

def main(model_path, time_limit):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"NNUE Chess - Playing against {model_path}")
    clock = pygame.time.Clock()
    
    bg, piece_images = load_assets()
    
    # Initialize Engine
    print("Loading Engine...")
    engine = NNUEEngine(model_path)
    print("Engine Loaded.")
    
    board = chess.Board()
    selected_square = None
    
    running = True
    game_over = False
    
    while running:
        clock.tick(FPS)
        
        # Check for Engine Turn
        if not game_over and board.turn == chess.BLACK:
            # Simple UI update before thinking
            draw_board(screen, bg)
            draw_pieces(screen, board, piece_images)
            # Add "Thinking..." overlay
            font = pygame.font.SysFont("Arial", 32)
            text = font.render(f"Engine Thinking ({time_limit}s)...", True, (255, 0, 0))
            screen.blit(text, (10, 10))
            pygame.display.flip()
            
            # Process events to prevent freeze cursor (not fully non-blocking but helps)
            pygame.event.pump()
            
            start = time.time()
            move = engine.search(board, time_limit=time_limit)
            if move:
                board.push(move)
                # Clear any events (clicks) that happened while engine was thinking
                pygame.event.clear()
            else:
                print("Engine resigns or bug.")
                game_over = True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over and board.turn == chess.WHITE:
                sq = get_square_under_mouse(event.pos)
                if sq is not None:
                    if selected_square is None:
                        # Select piece
                        if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE:
                             selected_square = sq
                    else:
                        # Target square
                        move = chess.Move(selected_square, sq)
                        # Check promotion
                        if board.piece_at(selected_square).piece_type == chess.PAWN and chess.square_rank(sq) == 7:
                             move = chess.Move(selected_square, sq, promotion=chess.QUEEN)
                             
                        if move in board.legal_moves:
                            board.push(move)
                            selected_square = None
                        elif sq == selected_square:
                            selected_square = None # Deselect
                        else:
                            # If clicked another own piece, select that instead
                            if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE:
                                selected_square = sq
                            else:
                                selected_square = None

        # Drawing
        draw_board(screen, bg)
        
        # Highlight Interaction
        if selected_square is not None:
            # Highlight selected
            s = pygame.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill(HIGHLIGHT)
            
            f = chess.square_file(selected_square)
            r = chess.square_rank(selected_square)
            screen.blit(s, (f*SQ_SIZE, (7-r)*SQ_SIZE))
            
            # Show legal moves
            for move in board.legal_moves:
                if move.from_square == selected_square:
                    to_sq = move.to_square
                    f2 = chess.square_file(to_sq)
                    r2 = chess.square_rank(to_sq)
                    pygame.draw.circle(screen, MOVE_HINT, (f2*SQ_SIZE + SQ_SIZE//2, (7-r2)*SQ_SIZE + SQ_SIZE//2), SQ_SIZE//6)

        draw_pieces(screen, board, piece_images)
        
        if board.is_game_over():
            game_over = True
            font = pygame.font.SysFont("Arial", 64)
            res = board.result()
            text = font.render(f"Game Over: {res}", True, (255, 0, 0))
            text_rect = text.get_rect(center=(WIDTH/2, HEIGHT/2))
            screen.blit(text, text_rect)
            
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints_1/checkpoint_latest.pth")
    parser.add_argument("--time", type=float, default=2.0)
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Warning: Model {args.model} not found. Ensure you have trained first.")
        
    main(args.model, args.time)
