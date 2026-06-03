import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
from tqdm import tqdm
from model import ChessNN, fen_to_tensor

class ChessDataset(Dataset):
    def __init__(self, data_path, device='cpu'):
        print(f"Loading data from {data_path}...")
        raw_data = torch.load(data_path) # List of (fen, score)
        self.data = raw_data
        self.device = device
        print(f"Loaded {len(self.data)} positions.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, score = self.data[idx]
        # On-the-fly conversion might be slow, but okay for MVP.
        # Pre-computing tensors is better for speed.
        x = fen_to_tensor(fen, device='cpu') # Keep on CPU for dataloader collate
        
        if " b " in fen:
             # FEN indicates Black to move.
             # Dataset score is "Relative" (PovScore).
             # So if Black is to move, +Score means Black is winning.
             # We want "White Advantage" (Absolute).
             # So we must negate the score.
             score = -score

        # Clamp score to eliminate huge mate values (e.g. +/- 10000) dominating the loss
        score = max(-2000, min(2000, score))
        
        # Normalize: Convert centipawns to "pawns" (divide by 100)
        norm_score = score / 100.0
        
        y = torch.tensor([norm_score], dtype=torch.float32)
        return x, y

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check for resuming
    start_epoch = 0
    model = ChessNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pth")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed at epoch {start_epoch}")

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    dataset = ChessDataset(args.data, device=device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # workers=0 for simplicity with simple objects

    criterion = nn.MSELoss()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        steps_in_epoch = 0
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps_in_epoch += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Step-wise saving? Maybe too frequent.
            # Save every 1000 steps?
            # User asked: "save the model at every few steps"
            if steps_in_epoch % 1000 == 0:
                 torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save Epoch Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        
        # Periodic history save
        if (epoch + 1) % 5 == 0:
             torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset.pt", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_1")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Dataset {args.data} not found. Please run data_gen.py first (requires Stockfish).")
    else:
        train(args)
