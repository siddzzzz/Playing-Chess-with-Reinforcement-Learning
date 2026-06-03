import sys
import chess
import chess.engine
import argparse
from engine import NNUEEngine
import threading
import time

# Helper to read stdin non-blocking (not strictly needed if we just block on input)
# But standard UCI loops usually just block on stdin.

class UCIAdapter:
    def __init__(self, model_path):
        self.engine = NNUEEngine(model_path)
        self.board = chess.Board()
        self.searching = False
        self.debug = False

    def log(self, msg):
        if self.debug:
            print(f"info string {msg}", flush=True)

    def run(self):
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                
                self.handle_command(line)
            except Exception as e:
                self.log(f"Error: {e}")

    def handle_command(self, cmd_line):
        parts = cmd_line.split()
        cmd = parts[0]

        if cmd == "uci":
            print("id name NNUE_AlphaBeta_v1")
            print("id author Antigravity")
            print("uciok", flush=True)

        elif cmd == "isready":
            print("readyok", flush=True)

        elif cmd == "ucinewgame":
            self.board = chess.Board()
            # Clear hash table if needed
            self.engine.transposition_table = {}

        elif cmd == "position":
            # position [fen <fenstring> | startpos] moves <move1> ...
            try:
                moves_idx = -1
                if "moves" in parts:
                    moves_idx = parts.index("moves")
                
                if parts[1] == "startpos":
                    self.board = chess.Board()
                    if moves_idx != -1:
                        moves = parts[moves_idx+1:]
                elif parts[1] == "fen":
                    # Reconstruct FEN
                    if moves_idx != -1:
                        fen_parts = parts[2:moves_idx]
                        moves = parts[moves_idx+1:]
                    else:
                        fen_parts = parts[2:]
                        moves = []
                    
                    fen = " ".join(fen_parts)
                    self.board = chess.Board(fen)
                
                if moves_idx != -1:
                    for move_uci in moves:
                        self.board.push_uci(move_uci)
                        
            except Exception as e:
                self.log(f"Position Error: {e}")

        elif cmd == "go":
            # go wtime 60000 btime 60000 winc 1000 binc 1000 depth 10...
            # Parse limits
            limits = {}
            for i in range(1, len(parts)):
                if parts[i] in ["wtime", "btime", "winc", "binc", "depth", "movetime"]:
                    try:
                        limits[parts[i]] = int(parts[i+1])
                    except:
                        pass
            
            self.start_search(limits)

        elif cmd == "stop":
            # Real engine would flag a stop variable. 
            # Since our search is blocking (single threaded in engine.py),
            # we can't really "stop" it gracefully unless we threaded it.
            # But for simple UCI, we often just wait for it to finish.
            pass

        elif cmd == "quit":
            sys.exit(0)

    def start_search(self, limits):
        # Determine strict time limit
        time_limit = 5.0 # default
        
        turn = self.board.turn # chess.WHITE or chess.BLACK
        
        # Calculate time management
        if "movetime" in limits:
            time_limit = limits["movetime"] / 1000.0
        elif "wtime" in limits and turn == chess.WHITE:
            t = limits["wtime"] / 1000.0
            inc = limits.get("winc", 0) / 1000.0
            # Simple Time Management: Use 1/20th of remaining time + inc/2
            time_limit = t / 30 + inc / 2
        elif "btime" in limits and turn == chess.BLACK:
            t = limits["btime"] / 1000.0
            inc = limits.get("binc", 0) / 1000.0
            time_limit = t / 30 + inc / 2
            
        # Minimum safety
        if time_limit < 0.05: time_limit = 0.05
        
        # self.log(f"Thinking for {time_limit:.2f}s")
        
        best_move = self.engine.search(self.board, time_limit=time_limit)
        
        if best_move:
            print(f"bestmove {best_move.uci()}", flush=True)
        else:
            # Should not happen with our fallback fix, but safety:
            print("bestmove 0000", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/checkpoint_latest.pth")
    args = parser.parse_args()
    
    adapter = UCIAdapter(args.model)
    adapter.run()
