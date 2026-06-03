import chess
import chess.svg
import time
from engine import NNUEEngine
import argparse
import os

def play_game(model_path, time_limit=2.0):
    engine = NNUEEngine(model_path)
    board = chess.Board()
    
    while not board.is_game_over():
        print(board)
        print(f"\nTurn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        if board.turn == chess.WHITE:
            # Human move
            while True:
                try:
                    move_str = input("Enter move (e.g., e2e4): ")
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move. Try again.")
                except ValueError:
                    print("Invalid format. Use UCI (e.g., e2e4).")
        else:
            # Engine move
            print("Engine thinking...")
            move = engine.search(board, time_limit=time_limit)
            if move:
                board.push(move)
                print(f"Engine play: {move}")
            else:
                print("Engine failed to find a move?")
                break
                
    print("Game Over")
    print(board.result())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/checkpoint_latest.pth")
    parser.add_argument("--time", type=float, default=2.0)
    args = parser.parse_args()
    
    play_game(args.model, args.time)
