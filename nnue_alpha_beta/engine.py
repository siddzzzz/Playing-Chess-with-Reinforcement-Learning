import chess
import torch
import time
from model import ChessNN, fen_to_tensor
import math

class NNUEEngine:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = ChessNN().to(device)
        self.model.eval()
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Support both raw state dict and checkpoint dict
            if 'model_state_dict' in checkpoint:
                 self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                 self.model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        except FileNotFoundError:
            print(f"Model not found at {model_path}, using random weights.")
        except Exception as e:
            print(f"Error loading model: {e}")

        self.transposition_table = {}
        self.nodes_visited = 0
        self.start_time = 0
        self.time_limit = 0
        self.last_search_info = {}
        self.history_table = {}

    def evaluate(self, board):
        # Convert board to tensor
        fen = board.fen()
        with torch.no_grad():
            x = fen_to_tensor(fen, device=self.device).unsqueeze(0) # Batch size 1
            # Model predicts in "pawns", convert back to centipawns
            score = self.model(x).item() * 100.0
        return score

    def search(self, board, time_limit=5.0):
        self.nodes_visited = 0
        self.start_time = time.time()
        self.time_limit = time_limit
        self.transposition_table = {} 
        self.last_search_info = {}
        self.history_table = {} # Reset history for fresh search
        
        best_move = None
        max_depth = 1
        
        print(f"Starting search for {time_limit}s...")
        
        while True:
            # Check time (only if we have at least one move)
            # We enforce that Depth 1 MUST complete.
            can_timeout = (max_depth > 1)
            
            if can_timeout and time.time() - self.start_time > self.time_limit:
                break
                
            try:
                score, move = self.alpha_beta_root(board, max_depth, -math.inf, math.inf, can_timeout)
                if move:
                    best_move = move
                    # Store info for GUI/Debug
                    self.last_search_info = {
                        'depth': max_depth,
                        'score': score,
                        'nodes': self.nodes_visited,
                        'pv': move
                    }
                    print(f"Depth {max_depth}: Move {move} Score {score:.2f} Nodes {self.nodes_visited}")
                max_depth += 1
            except TimeoutError:
                break
                
        return best_move

    def alpha_beta_root(self, board, depth, alpha, beta, can_timeout=True):
        best_move = None
        best_score = -math.inf
        
        moves = list(board.legal_moves)
        
        # Move Ordering: Search captures first for better pruning
        # Simple heuristic: Captures > Non-captures
        # Even better (MVV-LVA): Victim value - Attacker value (but simple capture check is good start)
        moves.sort(key=lambda m: board.is_capture(m), reverse=True)
        
        for move in moves:
            board.push(move)
            try:
                score = -self.alpha_beta(board, depth - 1, -beta, -alpha, can_timeout)
            finally:
                board.pop()
            
            if can_timeout and time.time() - self.start_time > self.time_limit:
                raise TimeoutError

            if score > best_score:
                best_score = score
                best_move = move
            
            if score > alpha:
                alpha = score
                
            # No beta cutoff at root
            
        return best_score, best_move

    def alpha_beta(self, board, depth, alpha, beta, can_timeout=True):
        self.nodes_visited += 1

        if can_timeout and self.nodes_visited % 1000 == 0:
             if time.time() - self.start_time > self.time_limit:
                 raise TimeoutError
        
        key = board.fen() 
        if key in self.transposition_table:
            entry = self.transposition_table[key]
            if entry['depth'] >= depth:
                return entry['score']

        if board.is_game_over():
            if board.is_checkmate(): 
                 return -10000 + (100-depth) 
            return 0 
            
        # Explicitly check for 3-fold repetition or 50-move rule draw claims
        if board.can_claim_draw():
            return 0 

        # Quiescence Search at leaf nodes
        if depth == 0:
            return self.quiescence(board, alpha, beta)

        best_score = -math.inf
        
        # Null Move Pruning
        if depth >= 3 and not board.is_check() and can_timeout:
             board.push(chess.Move.null())
             try:
                 score = -self.alpha_beta(board, depth - 1 - 2, -beta, -beta + 1, can_timeout)
             finally:
                 board.pop()
             
             if score >= beta:
                 return beta

        moves = list(board.legal_moves)
        
        # --- Move Ordering ---
        # 1. Captures (MVV-LVA logic simplified: just is_capture)
        # 2. Quiet Moves sorted by History Heuristic
        
        def move_score(m):
            if board.is_capture(m):
                return 1000000 # Captures always first
            return self.history_table.get(m, 0)
        
        moves.sort(key=move_score, reverse=True)
        # ---------------------
        
        for move in moves:
            board.push(move)
            try:
                score = -self.alpha_beta(board, depth - 1, -beta, -alpha, can_timeout)
            finally:
                board.pop()
            
            if can_timeout and time.time() - self.start_time > self.time_limit:
                raise TimeoutError

            if score > best_score:
                best_score = score
            
            if score > alpha:
                alpha = score
                # Update History Heuristic for quiet moves that cause cutoff
                if not board.is_capture(move):
                    self.history_table[move] = self.history_table.get(move, 0) + depth * depth
                
            if alpha >= beta:
                break
        
        self.transposition_table[key] = {'depth': depth, 'score': best_score}
        return best_score

    def quiescence(self, board, alpha, beta):
        self.nodes_visited += 1
        
        # Stand-pat score
        stand_pat = self.evaluate(board)
        if board.turn == chess.BLACK:
            stand_pat = -stand_pat

        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Filter captures only
        moves = list(board.generate_legal_moves(chess.BB_ALL, chess.BB_ALL)) 
        
        for move in moves:
            if not board.is_capture(move):
                continue
                
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha
