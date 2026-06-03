import torch
import chess
import chess.svg
import numpy as np
from model import ChessNet
from mcts import MCTS
import os

def play_game():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load Model
    model = ChessNet().to(device)
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", weights_only=True))
        print("Loaded best_model.pth")
    else:
        print("No model found, playing with random weights (very bad!)")
        
    model.eval()
    mcts = MCTS(model, device)
    board = chess.Board()
    
    # Game Loop
    while not board.is_game_over():
        print(f"\n{board}")
        
        if board.turn == chess.WHITE:
            # Human (White)
            while True:
                move_str = input("Enter your move (e.g. e2e4): ")
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move, try again.")
                except:
                    print("Invalid format, use UCI (e.g. e2e4).")
        else:
            # AI (Black)
            print("AI is thinking...")
            root = mcts.search(board, simulations=100) # Higher simulations for play
            moves, probs = mcts.get_action_probs(root, temperature=0) # Deterministic
            
            # Choose best
            best_idx = np.argmax(probs)
            move_idx = moves[best_idx]
            
            # Find move object
            # (Copy paste decode logic or loop legal moves)
            from utils import encode_move
            found_move = None
            for m in board.legal_moves:
                if encode_move(m) == move_idx:
                    found_move = m
                    break
            
            if found_move:
                print(f"AI plays: {found_move}")
                board.push(found_move)
            else:
                print("AI resigned (no valid move found in MCTS?)")
                break
                
    print("Game Over")
    print(board.result())

if __name__ == "__main__":
    play_game()
