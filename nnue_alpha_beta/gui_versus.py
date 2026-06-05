import pygame
import chess
import os
import sys
import time
from engine import NNUEEngine
import argparse
from gui_play import load_assets, draw_board, draw_pieces, WIDTH, HEIGHT, SQ_SIZE, FPS

def render_info(screen, game_num, total_games, p1_name, p2_name, score_p1, score_p2, current_turn_text):
    font = pygame.font.SysFont("Arial", 28)
    
    # Info Text
    text_lines = [
        f"Match: {game_num}/{total_games}",
        f"{p1_name} (White) vs {p2_name} (Black)" if game_num % 2 != 0 else f"{p2_name} (White) vs {p1_name} (Black)",
        f"Score: {p1_name} {score_p1} - {score_p2} {p2_name}",
        f"Turn: {current_turn_text}"
    ]
    
    y = 10
    for line in text_lines:
        text = font.render(line, True, (0, 0, 0))
        # Background box
        s = pygame.Surface(text.get_size())
        s.fill((255, 255, 255))
        s.set_alpha(220)
        screen.blit(s, (10, y))
        screen.blit(text, (10, y))
        y += 35

def main(old_model_path, new_model_path, unique_games=4, time_limit=0.5):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"NNUE Versus Mode: Old vs New")
    clock = pygame.time.Clock()
    bg, piece_images = load_assets()
    
    print("Loading Old Model...")
    engine_old = NNUEEngine(old_model_path)
    print("Loading New Model...")
    engine_new = NNUEEngine(new_model_path)
    
    # Player 1 = Old Model, Player 2 = New Model
    score_old = 0.0
    score_new = 0.0
    
    running = True
    
    for game_idx in range(1, unique_games + 1):
        if not running: break
        
        board = chess.Board()
        
        # Odd games (1, 3): Old=White, New=Black
        # Even games (2, 4): New=White, Old=Black
        old_is_white = (game_idx % 2 != 0)
        
        white_engine = engine_old if old_is_white else engine_new
        black_engine = engine_new if old_is_white else engine_old
        
        white_name = "Old Model" if old_is_white else "New Model"
        black_name = "New Model" if old_is_white else "Old Model"
        
        print(f"\n--- Game {game_idx}: {white_name} (White) vs {black_name} (Black) ---")
        
        while not board.is_game_over() and running:
            clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if not running: break
            
            draw_board(screen, bg)
            draw_pieces(screen, board, piece_images)
            
            turn_text = "White Thinking..." if board.turn == chess.WHITE else "Black Thinking..."
            render_info(screen, game_idx, unique_games, "Old", "New", score_old, score_new, turn_text)
            
            pygame.display.flip()
            pygame.event.pump()
            
            # Make Move
            engine_to_move = white_engine if board.turn == chess.WHITE else black_engine
            move = engine_to_move.search(board, time_limit=time_limit)
            
            if move:
                board.push(move)
            else:
                print("Engine Resigned/Crashed")
                break
        
        # Game Over
        if running:
            res = board.result()
            print(f"Game Over: {res}")
            
            # Update Score
            if res == "1-0":
                if old_is_white: score_old += 1
                else: score_new += 1
            elif res == "0-1":
                if not old_is_white: score_old += 1
                else: score_new += 1
            elif res == "1/2-1/2":
                score_old += 0.5
                score_new += 0.5
                
            # Show Result Screen for a few seconds
            for _ in range(60): # ~2 seconds at 30fps
                if not running: break
                draw_board(screen, bg)
                draw_pieces(screen, board, piece_images)
                render_info(screen, game_idx, unique_games, "Old", "New", score_old, score_new, f"Game Over: {res}")
                pygame.display.flip()
                clock.tick(30)
                pygame.event.pump()

    print(f"\n--- Match Finished ---")
    print(f"Final Score: Old Model {score_old} - {score_new} New Model")
    
    # Wait before closing
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        draw_board(screen, bg)
        font = pygame.font.SysFont("Arial", 48)
        text = font.render(f"Final: Old {score_old} - New {score_new}", True, (255, 0, 0))
        screen.blit(text, (WIDTH//2 - 200, HEIGHT//2))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=str, default="checkpoints/checkpoint_latest.pth")
    parser.add_argument("--new", type=str, default="checkpoints_1/checkpoint_latest.pth")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--time", type=float, default=0.5)
    args = parser.parse_args()
    
    main(args.old, args.new, args.games, args.time)
