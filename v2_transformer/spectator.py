import pygame
import chess
import time
import os
import sys

# Constants matching gui.py
WIDTH, HEIGHT = 600, 600
SQ_SIZE = WIDTH // 8
FPS = 30
ASSETS_DIR = "../assets" 

# Colors
WHITE_COL = (240, 217, 181)
BLACK_COL = (181, 136, 99)
HIGHLIGHT = (100, 255, 100, 100) # RGBA not supported by simple rect, but we can fix
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_TEXT_BLACK = (0, 0, 0)
HIGHLIGHT_COL = (255, 255, 0)

# Global for images
IMAGES = {}

def load_images():
    global IMAGES
    pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for p in pieces:
        path = os.path.join(ASSETS_DIR, f"{p}.png")
        try:
            if os.path.exists(path):
                img = pygame.image.load(path)
                IMAGES[p] = pygame.transform.smoothscale(img, (SQ_SIZE, SQ_SIZE))
            else:
                print(f"Missing asset: {path}")
        except Exception as e:
            print(f"Error loading {p}: {e}")

def draw_board(screen, board, last_move=None):
    # Draw Squares
    for r in range(8):
        for c in range(8):
            color = WHITE_COL if (r + c) % 2 == 0 else BLACK_COL
            pygame.draw.rect(screen, color, (c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
            
            # Highlight last move (simple rect)
            if last_move:
                # Board rank 0 = UI row 7
                # Board file 0 = UI col 0
                if last_move.from_square == chess.square(c, 7-r) or last_move.to_square == chess.square(c, 7-r):
                     s = pygame.Surface((SQ_SIZE, SQ_SIZE))
                     s.set_alpha(100)
                     s.fill(HIGHLIGHT_COL)
                     screen.blit(s, (c*SQ_SIZE, r*SQ_SIZE))

    # Draw Pieces
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            # Construct key e.g. 'wP'
            color_prefix = 'w' if piece.color == chess.WHITE else 'b'
            symbol_map = {'P':'P', 'N':'N', 'B':'B', 'R':'R', 'Q':'Q', 'K':'K'}
            # piece.symbol() returns lowercase for black but we want upper for type
            key = f"{color_prefix}{symbol_map[piece.symbol().upper()]}"
            
            if key in IMAGES:
                # Coords
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                r = 7 - rank
                c = file
                
                x = c * SQ_SIZE
                y = r * SQ_SIZE
                
                screen.blit(IMAGES[key], (x, y))
            else:
                # Fallback to circle
                pass

def draw_bar(screen, eval_score):
    # eval_score is approx [-1, 1]
    # -1 (Black Winning) -> Bar Empty (Black)
    # +1 (White Winning) -> Bar Full (White)
    
    # Clamp
    score = max(-1.0, min(1.0, eval_score))
    
    # Map [-1, 1] to [0, 1] height ratio
    ratio = (score + 1) / 2
    
    bar_width = 40
    bar_height = HEIGHT
    
    # X should be at the original WIDTH (600), shifting into the new space
    x = WIDTH
    y = 0
    
    # Background (Black/Gray)
    pygame.draw.rect(screen, (50, 50, 50), (x, y, bar_width, bar_height))
    
    # White part
    white_h = int(bar_height * ratio)
    pygame.draw.rect(screen, (220, 220, 220), (x, HEIGHT - white_h, bar_width, white_h))
    
    # Center Line
    pygame.draw.line(screen, (255, 0, 0), (x, HEIGHT//2), (x + bar_width, HEIGHT//2), 2)
    
    # Text
    font = pygame.font.SysFont('Arial', 16, bold=True)
    text = f"{eval_score:.2f}"
    
    # Choose text color based on background (Contrast)
    # If ratio > 0.5 (White bg), use Black text. Else White text.
    # Actually just put it in the middle or top/bottom?
    # Let's put it in the center.
    text_col = (0, 0, 0) if ratio > 0.5 else (255, 255, 255)
    
    # Shadow for readability
    ts = font.render(text, True, (128, 128, 128))
    t = font.render(text, True, text_col)
    
    tx = x + (bar_width - t.get_width()) // 2
    ty = HEIGHT // 2 - t.get_height() // 2
    
    screen.blit(ts, (tx+1, ty+1))
    screen.blit(t, (tx, ty))

def main():
    pygame.init()
    pygame.font.init()
    
    # Widen screen for bar
    screen = pygame.display.set_mode((WIDTH + 40, HEIGHT)) 
    pygame.display.set_caption("ChessGPT Live Spectator")
    clock = pygame.time.Clock()
    
    load_images()
    if not IMAGES:
        print("Warning: No images loaded. Board will be empty.")
    
    last_fen = ""
    board = chess.Board()
    eval_score = 0.0
    
    while True:
        try:
            if os.path.exists("live.fen"):
                with open("live.fen", "r") as f:
                    content = f.read().splitlines()
                    if content:
                         fen = content[0]
                         last_move_uci = content[1] if len(content) > 1 and content[1] != "None" else None
                         # Read Eval if available
                         if len(content) > 2:
                             try:
                                eval_score = float(content[2])
                             except: pass
                             
                         if fen != last_fen:
                             board.set_fen(fen)
                             last_fen = fen
                             
                             screen.fill((30, 30, 30)) # Clear bg
                             # Draw
                             last_move = chess.Move.from_uci(last_move_uci) if last_move_uci else None
                             draw_board(screen, board, last_move)
                             draw_bar(screen, eval_score)
                             pygame.display.flip()
        except Exception:
            pass # Ignore read errors (collision)
            
        for e in pygame.event.get():
             if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        clock.tick(30) # Poll at 30 FPS check

if __name__ == "__main__":
    main()
