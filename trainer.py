import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import chess
from tqdm import tqdm
import os
import random

from model import ChessNet
from mcts import MCTS
from utils import board_to_tensor, encode_move

class ChessDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # example: (state_tensor, policy_target_vector, value_target)
        # Note: policy_target needs to be expanded to full 4096 size if stored sparsely
        state, policy_dict, value = self.examples[idx]
        
        policy_target = np.zeros(4096, dtype=np.float32)
        for move_idx, prob in policy_dict.items():
            if move_idx < 4096:
                policy_target[move_idx] = prob
                
        return state, policy_target, np.array([value], dtype=np.float32)

class Trainer:
    def __init__(self, device=None):
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        print(f"Trainer using device: {self.device}")
        if self.device == 'cpu':
            print("WARNING: Training on CPU will be extremely slow. Please install PyTorch with CUDA support for your RTX 3050.")
            print("Command: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            
        self.model = ChessNet().to(self.device)
        try:
            self.model.load_state_dict(torch.load("best_model.pth", weights_only=True))
            print("Resuming training from best_model.pth")
        except:
            print("Starting fresh training.")

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # --- CONFIGURATION (Sequential / Local) ---
        self.mcts_sims = 100       
        self.games_per_iteration = 50 
        self.epochs = 10          
        self.batch_size = 64      
        # ------------------------------------------
        
    def execute_episode(self, game_idx=0, total_games=0):
        # Plays one game of self-play
        board = chess.Board()
        examples = []
        mcts = MCTS(self.model, self.device)
        
        while not board.is_game_over():
            # Print progress every move to reassure user
            if len(board.move_stack) % 1 == 0: # Every move
                print(f"\rGame {game_idx}/{total_games} | Move {len(board.move_stack)} | Simulating...", end="", flush=True)

            root = mcts.search(board, simulations=self.mcts_sims)
            
            # Use temperature=1 for first 30 moves, then 0 (deterministic) for exploration/exploitation balance
            temp = 1.0 if len(board.move_stack) < 30 else 0.1 # Reduced temp
            
            moves, probs = mcts.get_action_probs(root, temperature=temp)
            
            # Store data
            # Store policy as dict to save space before transforming
            policy_dict = {m: p for m, p in zip(moves, probs)}
            
            # Store state from current player perspective
            state_tensor = board_to_tensor(board)
            examples.append([state_tensor, policy_dict, None]) # Value filled later
            
            # Pick move
            # Sample from probs
            choice_idx = np.random.choice(len(moves), p=probs)
            best_move_idx = moves[choice_idx]
            
            # Decode and push
            # We need to find the move object from the root children to be safe or decode
            # But MCTS keys are indices.
            # We need to convert index back to move. But simpler: look at board.legal_moves
            # MCTS stores moves as indices.
            found_move = None
            for m in board.legal_moves:
                if encode_move(m) == best_move_idx:
                    found_move = m
                    break
            
            if found_move:
                board.push(found_move)
            else:
                print("Error: Move not found!")
                break
        
        # Clear the progress line
        print(f"\rGame {game_idx}/{total_games} Finished. Result: {board.result()}                  ")
                
        # Assignment of rewards
        result = board.result()
        if result == '1-0':
            reward = 1
        elif result == '0-1':
            reward = -1
        else:
            reward = 0
            
        # Backfill rewards
        # example list had [state, policy, None]
        # Iterate backwards
        # If White won (reward=1), then last state was Black moved? No.
        # State stored was "Player to move".
        # If State 0 was White to move. Winner is White. State 0 value = 1.
        # State 1 was Black to move. Winner is White. State 1 value = -1.
        
        processed_examples = []
        current_reward = reward # Reward for White from absolute perspective
        
        # We need to know who was to play in each state.
        # We can deduce from index? 0=White, 1=Black...
        # Or just use the alternating nature.
        
        # Actually easier: The reward `current_reward` is for White.
        # For each example, if player was White, target is `current_reward`. If Black, target is `-current_reward`.
        
        # Helper to know turn:
        # We can't easily know turn from just the stored list unless we tracked it.
        # But we know it alternates. Start is White.
        
        for i, ex in enumerate(examples):
            # i=0 -> White, i=1 -> Black
            perspective = 1 if (i % 2 == 0) else -1
            value_target = current_reward * perspective
            ex[2] = value_target
            processed_examples.append(ex)
            
        return processed_examples

    def train(self, iterations=1):
        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}...")
            # 1. Self Play
            iteration_examples = []
            # Let's use a simple range loop to avoid tqdm spam conflict
            for g in range(self.games_per_iteration):
                iteration_examples.extend(self.execute_episode(g+1, self.games_per_iteration))
            
            # 2. Train
            print(f"Training on {len(iteration_examples)} positions...")
            dataset = ChessDataset(iteration_examples)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            self.model.train()
            total_loss = 0
            for state, p_target, v_target in tqdm(dataloader, desc="Training"):
                state, p_target, v_target = state.to(self.device), p_target.to(self.device), v_target.to(self.device)
                
                self.optimizer.zero_grad()
                p_out, v_out = self.model(state)
                
                # Loss
                # Value: MSE
                v_loss = F.mse_loss(v_out.view(-1), v_target.view(-1))
                
                # Policy: Cross Entropy
                # p_out is logits. p_target is probs.
                # We can use log_softmax + nll_loss or specialized CrossEntropy functions.
                # F.cross_entropy expects class indices, but we have soft targets (probs).
                # So we use: - sum(target * log(softmax(input)))
                
                log_probs = F.log_softmax(p_out, dim=1)
                p_loss = -torch.sum(p_target * log_probs) / state.size(0)
                
                loss = v_loss + p_loss
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            print(f"Avg Loss: {total_loss / len(dataloader):.4f}")
            
            # Save Latest
            torch.save(self.model.state_dict(), "best_model.pth")
            
            # Save Checkpoint (History)
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(self.model.state_dict(), f"checkpoints/model_iter_{i+1}.pth")
            print(f"Model saved: best_model.pth & checkpoints/model_iter_{i+1}.pth")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(iterations=1)
