import math
import numpy as np
import torch
import chess
from utils import board_to_sequence, encode_move, CONTEXT_SIZE, get_material_score

class MCTSNode:
    def __init__(self, board, parent=None, prior=0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.is_expanded = False
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, device='cuda', c_puct=1.0):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        
    def search(self, board, simulations=50):
        root = MCTSNode(board.copy(), prior=0)
        
        # Expand root
        self._expand(root)
        
        for _ in range(simulations):
            node = root
            while node.is_expanded and len(node.children) > 0:
                node = self._select_child(node)
                
            value = self._expand(node)
            self._backpropagate(node, value)
            
        return root

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        total_visits = sum(child.visit_count for child in node.children.values())
        sqrt_total_visits = math.sqrt(total_visits)
        
        for move_idx, child in node.children.items():
            q_value = -child.value()
            u_value = self.c_puct * child.prior * sqrt_total_visits / (1 + child.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _expand(self, node):
        board = node.board
        if board.is_game_over():
            result = board.result()
            if result == '1-0': return 1 if board.turn == chess.WHITE else -1
            elif result == '0-1': return -1 if board.turn == chess.WHITE else 1
            else: return 0

        # TRANSFORMER INPUT
        # Convert board -> Sequence (64 tokens)
        sequence = board_to_sequence(board)
        seq_tensor = torch.tensor([sequence], dtype=torch.long, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(seq_tensor)
            
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        # HYBRID EVALUATION: Blend Learned Value with Material Heuristic
        # This helps the model "know" about material immediately during the game
        # without waiting for thousands of games to learn it.
        
        nn_value = value.item() # [-1, 1], Relative to Current Player
        
        # Calculate Material Score (e.g. +3, -5)
        # get_material_score returns (White - Black)
        mat_score = get_material_score(board) 
        
        # CRITICAL FIX: MCTS expects value "For Current Player".
        # If I am Black, positive material for White is BAD for me.
        if board.turn == chess.BLACK:
            mat_score = -mat_score
            
        mat_value = np.tanh(mat_score / 3.0) # Relative Material Advantage
        
        # Weighted mix: 30% Brain, 70% Material Rule
        # We assume the untrained brain is noisy, so we trust Material more initially.
        nodes_value = (0.3 * nn_value) + (0.7 * mat_value)
        
        policy_sum = 0
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            move_idx = encode_move(move)
            if move_idx < 4096:
                prob = policy_probs[move_idx]
                policy_sum += prob
                next_board = board.copy()
                next_board.push(move)
                child = MCTSNode(next_board, parent=node, prior=prob)
                node.children[move_idx] = child
        
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
            value = -value

    def get_action_probs(self, root, temperature=1.0):
        visits = [child.visit_count for child in root.children.values()]
        moves = [move_idx for move_idx in root.children.keys()]
        if sum(visits) == 0: return [], []
        
        if temperature == 0:
            best_idx = np.argmax(visits)
            probs = [0] * len(visits)
            probs[best_idx] = 1
            return moves, probs
        
        scaled_visits = [v ** (1.0 / temperature) for v in visits]
        sum_scaled = sum(scaled_visits)
        probs = [v / sum_scaled for v in scaled_visits]
        return moves, probs
