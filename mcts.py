import math
import numpy as np
import torch
import chess
from utils import board_to_tensor, encode_move, decode_move

class MCTSNode:
    def __init__(self, board, parent=None, prior=0):
        self.board = board
        self.parent = parent
        self.children = {} # Map move_index -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.is_expanded = False
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def ucb_score(self, child_prior, child_visits, c_puct=1.0):
        # PUCT formula
        # Q + U
        # U = c_puct * P * sqrt(N_parent) / (1 + N_child)
        q_value = 0
        if child_visits > 0:
            # We don't have direct access to child value here without looking up child node
            # But the caller will provide Q usually or we get it from child node
            pass 
        return 0 # Placeholder, logic is in select_child

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class MCTS:
    def __init__(self, model, device='cuda', c_puct=1.0):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        
    def search(self, board, simulations=50):
        # Create root
        root = MCTSNode(board.copy(), prior=0)
        
        # Expand root first to get priors
        self._expand(root)
        
        for _ in range(simulations):
            node = root
            
            # Select
            while node.is_expanded and len(node.children) > 0:
                node = self._select_child(node)
                
            # Expand & Evaluate
            value = self._expand(node)
            
            # Backpropagate
            self._backpropagate(node, value)
            
        return root

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        
        # Total visits for parent
        total_visits = sum(child.visit_count for child in node.children.values())
        sqrt_total_visits = math.sqrt(total_visits)
        
        for move_idx, child in node.children.items():
            q_value = -child.value() # Value is from perspective of the player moving into that state. 
                                     # child.value() is value for the player WHO JUST MOVED (the opponent of current node)
                                     # So we negate it? 
                                     # Let's clarify:
                                     # Value Head returns value for the CURRENT player in that state.
                                     # Board state passed to NN is "Player to move".
                                     # Value [1] means Player to move wins.
                                     # If I make move M, resulting state S' is Opponent to move.
                                     # NN(S') gives Value' (for Opponent).
                                     # So My Value = -Value'.
                                     
            
            u_value = self.c_puct * child.prior * sqrt_total_visits / (1 + child.visit_count)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    def _expand(self, node):
        board = node.board
        
        # Terminal checks
        if board.is_game_over():
            result = board.result()
            if result == '1-0':
                return 1 if board.turn == chess.WHITE else -1
            elif result == '0-1':
                return -1 if board.turn == chess.WHITE else 1
            else:
                return 0 # Draw
        
        # Prepare input
        state_tensor = board_to_tensor(board)
        state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        nodes_value = value.item() # Scalar [-1, 1] for current player
        
        # Create children for legal moves
        legal_moves = list(board.legal_moves)
        
        # Mask illegal moves and normalize policy manually 
        # (Usually AlphaZero masks before softmax, but masking after is OK for simple implementation)
        
        policy_sum = 0
        for move in legal_moves:
            move_idx = encode_move(move)
            if move_idx < 4096:
                prob = policy_probs[move_idx]
                policy_sum += prob
                
                # Create Child
                next_board = board.copy()
                next_board.push(move)
                child = MCTSNode(next_board, parent=node, prior=prob)
                node.children[move_idx] = child
        
        # Re-normalize priors
        if policy_sum > 0:
            for child in node.children.values():
                child.prior /= policy_sum
                
        node.is_expanded = True
        return nodes_value

    def _backpropagate(self, node, value):
        curr = node
        while curr is not None:
            curr.visit_count += 1
            curr.value_sum += value
            curr = curr.parent
            value = -value # Flip perspective at each level
            
    def get_action_probs(self, root, temperature=1.0):
        visits = [child.visit_count for child in root.children.values()]
        moves = [move_idx for move_idx in root.children.keys()]
        
        if sum(visits) == 0:
            return [], [] # Should not happen if simulations > 0

        if temperature == 0:
            # Deterministic: pick max visits
            best_idx = np.argmax(visits)
            probs = [0] * len(visits)
            probs[best_idx] = 1
            return moves, probs
        
        # Softmax with temp
        # visits^(1/temp)
        scaled_visits = [v ** (1.0 / temperature) for v in visits]
        sum_scaled = sum(scaled_visits)
        probs = [v / sum_scaled for v in scaled_visits]
        
        return moves, probs
