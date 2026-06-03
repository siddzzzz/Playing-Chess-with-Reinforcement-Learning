import pygame
import chess
import math
import sys
import torch
import numpy as np
import threading
import time
from model import ChessNet
from mcts import MCTS
from utils import encode_move

# Configuration
WIDTH, HEIGHT = 800, 800 # Window size
BOARD_SIZE = 640 # Board size
SQUARE_SIZE = BOARD_SIZE // 8
OFFSET_X = (WIDTH - BOARD_SIZE) // 2
OFFSET_Y = (HEIGHT - BOARD_SIZE) // 2

# Colors
COLOR_BG = (30, 30, 30)
COLOR_BOARD_LIGHT = (240, 217, 181)
COLOR_BOARD_DARK = (181, 136, 99)
COLOR_HIGHLIGHT = (100, 255, 100, 100) # RGBA
COLOR_LAST_MOVE = (255, 255, 0, 100)
COLOR_TEXT_WHITE = (255, 255, 255)
COLOR_TEXT_BLACK = (0, 0, 0)

# Init Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AlphaZero Chess (RTX 3050)")
font = pygame.font.SysFont("segoe ui symbol", int(SQUARE_SIZE * 0.8))
ui_font = pygame.font.SysFont("arial", 24)

# Global State
board = chess.Board()
ai_thinking = False
game_over = False
selected_square = None
legal_moves = []
last_move = None
ai_move_found = None

# Piece Unicode Map
PIECES = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}

def load_assets():
    bg = None
    try:
        bg = pygame.image.load("assets/chess_background.png")
        bg = pygame.transform.smoothscale(bg, (WIDTH, HEIGHT))
        # Darken it a bit
        dark = pygame.Surface((WIDTH, HEIGHT)).convert_alpha()
        dark.fill((0, 0, 0, 150))
        bg.blit(dark, (0, 0))
    except:
        pass
    return bg

BACKGROUND = load_assets()

def draw_board(screen):
    if BACKGROUND:
        screen.blit(BACKGROUND, (0, 0))
    else:
        screen.fill(COLOR_BG)
        
    # Draw Squares
    for r in range(8):
        for c in range(8):
            color = COLOR_BOARD_LIGHT if (r + c) % 2 == 0 else COLOR_BOARD_DARK
            x = OFFSET_X + c * SQUARE_SIZE
            y = OFFSET_Y + r * SQUARE_SIZE # Rank 0 is bottom in chess?
            # pygame 0,0 is top left. 
            # chess board default: rank 7 is top, rank 0 is bottom.
            # So row 0 in UI is rank 7. row 7 in UI is rank 0.
            # Let's map visual row to rank: rank = 7 - r
            
            rect = (x, y, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)
            
            # Highlight Last Move
            if last_move:
                # from
                if last_move.from_square == chess.square(c, 7-r) or last_move.to_square == chess.square(c, 7-r):
                    s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                    s.set_alpha(100)
                    s.fill((255, 255, 0))
                    screen.blit(s, (x, y))

            # Highlight Selected
            if selected_square is not None and selected_square == chess.square(c, 7-r):
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                s.set_alpha(150)
                s.fill((0, 255, 0))
                screen.blit(s, (x, y))
                
    # Draw Pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # Map to UI
            # rank 0 -> r=7, rank 7 -> r=0
            r = 7 - rank
            c = file
            
            x = OFFSET_X + c * SQUARE_SIZE
            y = OFFSET_Y + r * SQUARE_SIZE
            
            # Render Text
            symbol = PIECES[piece.symbol()]
            color = COLOR_TEXT_WHITE if piece.color == chess.WHITE else COLOR_TEXT_BLACK
            
            # Shadow
            text_s = font.render(symbol, True, (100, 100, 100))
            text_r = text_s.get_rect(center=(x + SQUARE_SIZE//2 + 2, y + SQUARE_SIZE//2 + 2))
            screen.blit(text_s, text_r)
            
            text = font.render(symbol, True, color)
            text_rect = text.get_rect(center=(x + SQUARE_SIZE//2, y + SQUARE_SIZE//2))
            screen.blit(text, text_rect)
            
    # Draw Legal Move Hints
    if selected_square is not None:
        for move in legal_moves:
            if move.from_square == selected_square:
                to_sq = move.to_square
                file = chess.square_file(to_sq)
                rank = chess.square_rank(to_sq)
                r = 7 - rank
                c = file
                x = OFFSET_X + c * SQUARE_SIZE + SQUARE_SIZE//2
                y = OFFSET_Y + r * SQUARE_SIZE + SQUARE_SIZE//2
                pygame.draw.circle(screen, (0, 200, 0), (x, y), 10)

def ai_worker(model, device, current_board):
    global ai_move_found, ai_thinking
    mcts = MCTS(model, device)
    # Copy board for thread safety
    b_copy = current_board.copy()
    
    # Run Search
    sims = 400 # Decent search
    root = mcts.search(b_copy, simulations=sims)
    moves, probs = mcts.get_action_probs(root, temperature=0)
    
    best_idx = np.argmax(probs)
    move_idx = moves[best_idx]
    
    found_move = None
    for m in b_copy.legal_moves:
        if encode_move(m) == move_idx:
            found_move = m
            break
            
    ai_move_found = found_move
    ai_thinking = False

def main():
    global board, selected_square, legal_moves, last_move, ai_thinking, ai_move_found, game_over
    
    # Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"GUI using device: {device}")
    model = ChessNet().to(device)
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
        print("Model loaded.")
    except:
        print("Using random model.")
    model.eval()
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over and not ai_thinking:
                if board.turn == chess.WHITE: # Human turn
                    mx, my = pygame.mouse.get_pos()
                    # Check if inside board
                    if OFFSET_X <= mx < OFFSET_X + BOARD_SIZE and OFFSET_Y <= my < OFFSET_Y + BOARD_SIZE:
                        # Convert to coords
                        c = (mx - OFFSET_X) // SQUARE_SIZE
                        r = (my - OFFSET_Y) // SQUARE_SIZE
                        rank = 7 - r
                        file = c
                        sq = chess.square(file, rank)
                        
                        # Logic
                        if selected_square is None:
                            p = board.piece_at(sq)
                            if p and p.color == chess.WHITE:
                                selected_square = sq
                                legal_moves = [m for m in board.legal_moves if m.from_square == sq]
                        else:
                            # Try to move
                            move = chess.Move(selected_square, sq)
                            # Check promotion (quick fix: always queen)
                            if board.piece_at(selected_square).piece_type == chess.PAWN and rank == 7:
                                move.promotion = chess.QUEEN
                            
                            if move in board.legal_moves:
                                board.push(move)
                                last_move = move
                                selected_square = None
                                legal_moves = []
                                # Trigger AI
                                ai_thinking = True
                                threading.Thread(target=ai_worker, args=(model, device, board)).start()
                            else:
                                # Reselect or Deselect
                                p = board.piece_at(sq)
                                if p and p.color == chess.WHITE:
                                    selected_square = sq
                                    legal_moves = [m for m in board.legal_moves if m.from_square == sq]
                                else:
                                    selected_square = None
                                    legal_moves = []

        # AI Move Application check (Main Thread)
        if ai_move_found and ai_thinking == False:
            board.push(ai_move_found)
            last_move = ai_move_found
            ai_move_found = None
            
        # Draw
        draw_board(screen)
        
        # UI Overlay
        if ai_thinking:
            status = ui_font.render("AI Thinking...", True, (255, 100, 100))
            screen.blit(status, (20, 20))
            
        if board.is_game_over():
            game_over = True
            res = board.result()
            txt = f"Game Over: {res}"
            t = ui_font.render(txt, True, (0, 255, 0))
            screen.blit(t, (WIDTH//2 - t.get_width()//2, HEIGHT//2))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
