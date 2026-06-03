import chess
import chess.engine
import torch
import random
import argparse
from tqdm import tqdm
import os

def generate_dataset(stockfish_path, num_positions, output_file, threads=1, depth=10):
    """
    Generates a dataset of unique chess positions and their Stockfish evaluations.
    """
    if not os.path.exists(stockfish_path):
        # check if it is in path
        import shutil
        if shutil.which(stockfish_path) is None:
             raise FileNotFoundError(f"Stockfish executable not found at {stockfish_path}")

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": threads})

    data = []
    board = chess.Board()
    
    seen_fens = set()
    
    print(f"Generating {num_positions} positions using {stockfish_path}...")
    
    with tqdm(total=num_positions) as pbar:
        while len(data) < num_positions:
            if board.is_game_over(claim_draw=True):
                board.reset()
            
            # "Guided" Self-Play Generation
            # 10% chance of random move (high exploration)
            # 90% chance of "decent" move (Stockfish Depth 1-2)
            if random.random() < 0.1:
                move = random.choice(list(board.legal_moves))
            else:
                # Ask engine for a quick move (Depth 2 is fast but avoids blunders)
                result = engine.play(board, chess.engine.Limit(depth=2))
                move = result.move

            board.push(move)
            
            fen = board.fen()
            
            # Simple check to avoid exact duplicates
            if fen in seen_fens:
                continue
            seen_fens.add(fen)
            
            # Evaluate (Ground Truth) - High Depth
            # We use a higher depth here to get the "Real" score of the position
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].relative.score(mate_score=10000)
            
            if score is None: 
                continue 
                
            data.append((fen, score))
            pbar.update(1)

            # Reset board if game ends or gets too long
            if board.is_game_over() or len(board.move_stack) > 150:
                 board.reset()
                 # Optional: Randomize start position slightly? 
                 # For now standard start is fine, random moves will diverge.

    engine.quit()
    
    # Save to pytorch file
    print(f"Saving dataset to {output_file}...")
    torch.save(data, output_file)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", type=str, default="stockfish", help="Path to stockfish executable")
    parser.add_argument("--num_positions", type=int, default=1000000, help="Number of positions to generate")
    parser.add_argument("--output", type=str, default="dataset.pt", help="Output file")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish analysis depth")
    
    args = parser.parse_args()
    
    try:
        generate_dataset(args.stockfish, args.num_positions, args.output, depth=args.depth)
    except Exception as e:
        print(f"Error: {e}")
