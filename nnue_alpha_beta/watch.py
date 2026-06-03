import chess
import time
from engine import NNUEEngine
import argparse
import os
import torch

def watch_self_play(checkpoint_dir, time_limit=1.0):
    last_load_time = 0
    board = chess.Board()
    engine_white = None
    engine_black = None
    
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    
    print(f"Watching training progress in {checkpoint_dir}...")
    
    while True:
        # Reload model if new checkpoint
        if os.path.exists(checkpoint_path):
            mod_time = os.path.getmtime(checkpoint_path)
            if mod_time > last_load_time:
                print("\nReloading updated model...")
                try:
                    # Re-instantiate to refresh
                    engine_white = NNUEEngine(checkpoint_path)
                    engine_black = NNUEEngine(checkpoint_path)
                    last_load_time = mod_time
                    # Optionally reset board on new model? 
                    # Or let the game finish?
                    # Let's let the game continue but with better brains.
                except Exception as e:
                    print(f"Read error (training writing?): {e}")
                    time.sleep(1)
                    continue
        
        if engine_white is None:
            print("Waiting for first checkpoint...")
            time.sleep(5)
            continue
            
        if board.is_game_over():
            print(f"Game Over: {board.result()}")
            print(board)
            time.sleep(5)
            board.reset()
            print("\nStarting new game...")
            
        # Play move
        if board.turn == chess.WHITE:
            move = engine_white.search(board, time_limit=time_limit)
        else:
            move = engine_black.search(board, time_limit=time_limit)
            
        if move:
            board.push(move)
            print(f"\n{board}")
        else:
            print("No move found, game ended?")
            board.reset()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="checkpoints")
    parser.add_argument("--time", type=float, default=0.5)
    args = parser.parse_args()
    
    watch_self_play(args.dir, args.time)
