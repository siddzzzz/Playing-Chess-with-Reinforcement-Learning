import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import chess
import chess.svg
from tqdm import tqdm
import os
import random

from model import ChessTransformer
from mcts import MCTS
from utils import board_to_sequence, encode_move, CONTEXT_SIZE, get_material_score

class ChessDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # example: (sequence, policy_dict, value)
        sequence, policy_dict, value = self.examples[idx]
        
        # Sequence is already list of 64 integers (from board_to_sequence)
        # No padding needed for fixed board size
        
        policy_target = np.zeros(4096, dtype=np.float32)
        for move_idx, prob in policy_dict.items():
            if move_idx < 4096:
                policy_target[move_idx] = prob
                
        return torch.tensor(sequence, dtype=torch.long), policy_target, np.array([value], dtype=np.float32)

class Trainer:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Transformer Trainer using: {self.device}")
        
        self.model = ChessTransformer().to(self.device)
        if os.path.exists("transformer_model.pth"):
            try:
                self.model.load_state_dict(torch.load("transformer_model.pth", weights_only=True))
                print("Loaded existing transformer model.")
            except:
                pass
                
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001) # Lower LR for Transformers
        self.mcts_sims = 200 # Increased to 200 for smarter play
        self.games_per_iteration = 10 # Increased to 10 for better stability
        self.batch_size = 32
        
        # Resume game count
        self.total_games_played = 0
        if os.path.exists("games.csv"):
            with open("games.csv", "r") as f:
                self.total_games_played = sum(1 for line in f) - 1 # Minus header
            if self.total_games_played < 0: self.total_games_played = 0
        print(f"Resuming from Game {self.total_games_played}")

    def execute_episode(self, game_idx=0, total_games=0):
        board = chess.Board()
        examples = []
        mcts = MCTS(self.model, self.device)
        
        # Try to clean up previous outputs if using IPython
        try:
            from IPython.display import clear_output, display, SVG
        except ImportError:
            pass

        while not board.is_game_over():
            if len(board.move_stack) > 150: break # Prevent infinite length
            
            # Save Live State for Spectator
            try:
                # Calculate Evaluation for the BAR
                current_seq = board_to_sequence(board)
                seq_tensor = torch.tensor([current_seq], dtype=torch.long, device=self.device)
                self.model.eval()
                with torch.no_grad():
                     _, val_tensor = self.model(seq_tensor)
                
                # Hybrid Eval for consistency with MCTS
                nn_val = val_tensor.item() # Relative (Good for mover)
                
                # We want the Bar to always show "White Advantage".
                # If it's Black's turn, nn_val=+1 means Black wins (White -1).
                if board.turn == chess.BLACK:
                    nn_val = -nn_val
                
                mat_score = get_material_score(board) # Absolute (White - Black)
                mat_val = np.tanh(mat_score / 3.0) 
                
                # 50/50 blend (or 30/70 to match training)
                eval_score = (0.3 * nn_val) + (0.7 * mat_val)
                
                with open("live.fen", "w") as f:
                    f.write(board.fen())
                    # Also write last move for highlighting
                    if board.move_stack:
                        f.write(f"\n{board.peek().uci()}")
                    else:
                        f.write("\nNone")
                        
                    # Write Eval Score (Line 3)
                    f.write(f"\n{eval_score:.4f}")
            except Exception as e:
                pass
            
            # VISUALIZATION
            # Only visualize every move (or every few moves)
            # VISUALIZATION
            # Only visualize every move
            # Strict check: Are we in a notebook (ZMQInteractiveShell)?
            try:
                # Check for explicit Notebook environment
                is_notebook = False
                try:
                    from IPython import get_ipython
                    if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
                        is_notebook = True
                except:
                    pass

                if is_notebook:
                    clear_output(wait=True)
                    display(SVG(chess.svg.board(board, size=300)))
                    print(f"Game {game_idx}/{total_games} | Move {len(board.move_stack)}")
                else:
                    # Console fallback
                    print(f"\rGame {game_idx}/{total_games} | Move {len(board.move_stack)}", end="", flush=True)
            except Exception:
                pass

            root = mcts.search(board, simulations=self.mcts_sims)
            # Temperature schedule: Explore for first 80 moves, then exploit
            # Changed 1.0 -> 0.7 to make play sharper and less random
            temp = 0.7 if len(board.move_stack) < 80 else 0.5

            moves, probs = mcts.get_action_probs(root, temperature=temp)
            
            policy_dict = {m: p for m, p in zip(moves, probs)}
            
            # Store SEQUENCE
            current_seq = board_to_sequence(board)
            examples.append([current_seq, policy_dict, None])
            
            choice_idx = np.random.choice(len(moves), p=probs)
            best_move_idx = moves[choice_idx]
            
            found_move = None
            for m in board.legal_moves:
                if encode_move(m) == best_move_idx:
                    found_move = m
                    break
            
            if found_move:
                board.push(found_move)
            else:
                break

        print(f"\rGame {game_idx}/{total_games} Finished. Result: {board.result()}                   ")
        
        # Save to CSV
        try:
            with open("games.csv", "a", encoding="utf-8") as f:
                # Format: GameID, Result, Moves(SAN)
                moves_san = [board.move_stack[i] for i in range(len(board.move_stack))]
                # Reconstruct SAN is hard without board replay, but board has move_stack. 
                # board.move_stack contains Move objects. Getting SAN requires board history.
                # Simplest is just UCI string space separated.
                moves_str = " ".join([m.uci() for m in board.move_stack])
                f.write(f"{game_idx},{board.result()},{moves_str}\n")
        except Exception as e:
            print(f"CSV Save Error: {e}")

        result = board.result()
        material_score = get_material_score(board) # positive means White has advantage
        
        # Base Reward
        if result == '1-0': 
            base_reward = 1.0
        elif result == '0-1': 
            base_reward = -1.0
        else: 
            # Draw Scenarios
            if board.can_claim_threefold_repetition():
                # Massive penalty for repetition to stop "cowardly" early draws
                base_reward = -0.5 
            else:
                # Standard draw penalty (Stalemate, insufficient material, 50-move)
                base_reward = -0.1 
            
        # Add small material incentive (e.g. 0.01 per pawn advantage)
        # We clamp it to avoid overshadowing the actual result
        # Max material diff is roughly 39. 39 * 0.01 = 0.39.
        # So a Win (1.0) is still better than capturing a Queen (0.09) but losing (-1).
        
        final_score_white = base_reward + (material_score * 0.01)
        
        # Clip to [-1, 1] range roughly, or just let it float. Tanh output is [-1,1].
        # Let's clip for stability? MCTS expects bounds usually.
        # But for AlphaZero MCTS, value is just expectation.
        # Let's keep it simple.
            
        processed = []
        for i, ex in enumerate(examples):
            # Perspective: 1 if White's turn, -1 if Black's turn
            # White's Move (Event 0, 2, 4...) -> Matches White's reward
            # Black's Move (Event 1, 3, 5...) -> Matches Black's reward (inverse of White)
            
            perspective = 1 if (i % 2 == 0) else -1
            
            # The value target should be from the perspective of the player who just moved?
            # No, usually it's "Value of the state for the current player".
            # If state is White to move, and White wins, Value = +1.
            # Ex[2] is the label for the board state `ex[0]`.
            # If ex[0] is Start Pos (White to move), and White eventually wins, Label = 1.2
            # If White wins, and it's Black's turn (perspective -1), Label = -1.2
            
            ex[2] = final_score_white * perspective
            processed.append(ex)
        return processed

    def train(self, iterations=1):
        for i in range(iterations):
            print(f"Iteration {i+1}")
            examples = []
            for g in range(self.games_per_iteration):
                current_game_idx = self.total_games_played + g + 1
                examples.extend(self.execute_episode(current_game_idx, self.games_per_iteration))
            
            self.total_games_played += self.games_per_iteration
                
            dataset = ChessDataset(examples)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            self.model.train()
            total_loss = 0
            for seq, p_target, v_target in tqdm(dataloader):
                seq, p_target, v_target = seq.to(self.device), p_target.to(self.device), v_target.to(self.device)
                
                self.optimizer.zero_grad()
                p_out, v_out = self.model(seq)
                
                v_loss = F.mse_loss(v_out.view(-1), v_target.view(-1))
                p_loss = -torch.sum(p_target * F.log_softmax(p_out, dim=1)) / seq.size(0)
                
                loss = v_loss + p_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            print(f"Avg Loss: {avg_loss:.4f}")
            
            # Save Loss to CSV
            try:
                with open("loss.csv", "a") as f:
                    f.write(f"{self.total_games_played},{avg_loss:.4f}\n")
            except:
                pass

            torch.save(self.model.state_dict(), "transformer_model.pth")

if __name__ == "__main__":
    t = Trainer()
    t.train(10)
