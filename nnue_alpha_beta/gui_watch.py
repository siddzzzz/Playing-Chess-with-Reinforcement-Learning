import pygame
import chess
import os
import sys
import time
from engine import NNUEEngine
import argparse
from gui_play import load_assets, draw_board, draw_pieces, WIDTH, HEIGHT, SQ_SIZE, FPS, HIGHLIGHT, MOVE_HINT
# Reusing constants and drawing functions from gui_play to ensure consistency

def main(checkpoint_dir, time_limit):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"NNUE Watch - Spectating Training in {checkpoint_dir}")
    clock = pygame.time.Clock()
    
    bg, piece_images = load_assets()
    
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    last_load_time = 0
    
    engine_white = None
    engine_black = None
    
    board = chess.Board()
    
    running = True
    
    while running:
        clock.tick(FPS)
        
        # 1. Handle Events (Quit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 2. Check for Model Updates
        if os.path.exists(checkpoint_path):
            try:
                mod_time = os.path.getmtime(checkpoint_path)
                if mod_time > last_load_time:
                    # Debounce slightly to ensure write is complete
                    time.sleep(0.1) 
                    print(f"Reloading model from {checkpoint_path}...")
                    engine_white = NNUEEngine(checkpoint_path)
                    engine_black = NNUEEngine(checkpoint_path)
                    last_load_time = mod_time
                    pygame.display.set_caption(f"NNUE Watch - Model Updated: {time.ctime(mod_time)}")
            except Exception as e:
                print(f"Error reloading model: {e}")
        
        # 3. Game Logic
        if engine_white is None:
            # Waiting for first model
            draw_board(screen, bg)
            font = pygame.font.SysFont("Arial", 32)
            text = font.render(f"Waiting for checkpoint...", True, (255, 0, 0))
            screen.blit(text, (WIDTH//2 - 150, HEIGHT//2))
        else:
            # Check Game Over
            if board.is_game_over():
                draw_board(screen, bg)
                draw_pieces(screen, board, piece_images)
                
                font = pygame.font.SysFont("Arial", 48)
                res = board.result()
                text = font.render(f"Game Over: {res}", True, (255, 0, 0))
                text_rect = text.get_rect(center=(WIDTH/2, HEIGHT/2))
                
                # Draw semi-transparent box behind text
                box = pygame.Surface((400, 100))
                box.set_alpha(200)
                box.fill((0,0,0))
                box_rect = box.get_rect(center=(WIDTH/2, HEIGHT/2))
                screen.blit(box, box_rect)
                screen.blit(text, text_rect)
                
                pygame.display.flip()
                
                # Wait 5 seconds then reset
                start_wait = time.time()
                while time.time() - start_wait < 5:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            break
                    if not running: break
                    clock.tick(10)
                
                if running:
                    board.reset()
                    print("Starting new game...")
                    continue
            
            # Make Move
            # To avoid freezing UI, we should ideally thread this or just accept small freeze
            # Since simple engine is fast-ish and this is "watch" mode, blocking is OK 
            # if we draw "Thinking" first.
            
            draw_board(screen, bg)
            draw_pieces(screen, board, piece_images)
            
            # Highlight last move
            if board.move_stack:
                last_move = board.peek()
                # Draw highlight (optional, can steal from gui_play logic if needed)
            
            pygame.display.flip()
            # Process events to keep window responsive-ish
            pygame.event.pump()
            
            if board.turn == chess.WHITE:
                move = engine_white.search(board, time_limit=time_limit)
            else:
                move = engine_black.search(board, time_limit=time_limit)
                
            if move:
                board.push(move)
            else:
                print("No move found?")
                board.reset()

        # 4. Render Final Frame of step
        draw_board(screen, bg)
        draw_pieces(screen, board, piece_images)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="checkpoints")
    parser.add_argument("--time", type=float, default=0.5)
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Warning: Checkpoint directory {args.dir} not found.")
        
    main(args.dir, args.time)
