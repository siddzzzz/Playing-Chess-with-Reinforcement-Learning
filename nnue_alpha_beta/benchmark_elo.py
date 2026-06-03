import chess
import chess.engine
import chess.pgn
import argparse
import os
import subprocess
import sys
from engine import NNUEEngine
import datetime

# Optional visualization imports
try:
    import pygame
    from gui_play import load_assets, draw_board, draw_pieces, WIDTH, HEIGHT, FPS
    VISUAL_AVAILABLE = True
except ImportError:
    VISUAL_AVAILABLE = False

BAYESELO = "bayeselo"
PGN_FILE = "benchmark_custom.pgn"

def render_game(screen, bg, piece_images, board, text_info=""):
    if not VISUAL_AVAILABLE or screen is None:
        return
        
    draw_board(screen, bg)
    draw_pieces(screen, board, piece_images)
    
    if text_info:
        font = pygame.font.SysFont("Arial", 24)
        text = font.render(text_info, True, (0, 0, 0)) # Black text
        # Draw background for text
        s = pygame.Surface(text.get_size())
        s.fill((255, 255, 255))
        s.set_alpha(200)
        screen.blit(s, (10, 10))
        screen.blit(text, (10, 10))
        
    pygame.display.flip()
    pygame.event.pump() # Process events

def run_bayeselo():
    """Runs BayesElo on the generated PGN file."""
    if subprocess.run([BAYESELO], input="quit", encoding="utf-8", capture_output=True, check=False).returncode != 0:
        print("[WARNING] BayesElo not found. Skipping rating calculation.")
        return

    print("\nRunning BayesElo Analysis...")
    script = f"""
readpgn {PGN_FILE}
elo
mm
ratings
x
quit
"""
    try:
        result = subprocess.run([BAYESELO], input=script, capture_output=True, text=True, check=True)
        print("\n--- BayesElo Ratings ---")
        print(result.stdout)
    except Exception as e:
        print(f"BayesElo Failed: {e}")

def play_single_game(nnue_engine, stockfish_path, elo, round_num, time_limit=0.1, visual=False):
    """
    Plays a single game against Stockfish at a specific ELO.
    """
    sf_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        sf_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    except:
        pass 
            
    screen = None
    bg = None
    piece_images = None
    
    if visual and VISUAL_AVAILABLE:
        if not pygame.get_init():
             pygame.init()
             
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(f"Round {round_num} vs Stockfish {elo}")
        bg, piece_images = load_assets()

    pgn_file_handle = open(PGN_FILE, "a")

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = f"Benchmark ELO {elo}"
    game.headers["Round"] = str(round_num)
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    
    # Alternate colors based on round number
    # Round 1: MyNNUE White (1%2 != 0) -> No, 1 is Odd. 
    # Let's say Round 1 (i=0): White. Round 2 (i=1): Black.
    nnue_color = chess.WHITE if round_num % 2 != 0 else chess.BLACK
    
    if nnue_color == chess.WHITE:
        game.headers["White"] = "MyNNUE"
        game.headers["Black"] = f"Stockfish_{elo}"
    else:
        game.headers["White"] = f"Stockfish_{elo}"
        game.headers["Black"] = "MyNNUE"

    node = game
    score = 0
    res_str = "*"
    
    while not board.is_game_over():
        if visual:
            if nnue_color == chess.WHITE:
                info = f"R{round_num}: White(My AI) vs Black(SF {elo})"
            else:
                info = f"R{round_num}: White(SF {elo}) vs Black(My AI)"
            
            if hasattr(nnue_engine, 'last_search_info') and nnue_engine.last_search_info:
                d = nnue_engine.last_search_info.get('depth', 0)
                s = nnue_engine.last_search_info.get('score', 0.0)
                info += f" | D:{d} Sc:{s:.2f}"
                
            render_game(screen, bg, piece_images, board, info)
        
        if board.turn == nnue_color:
            move = nnue_engine.search(board, time_limit=time_limit)
            if move is None:
                res_str = "0-1" if nnue_color == chess.WHITE else "1-0" 
                break 
            board.push(move)
            node = node.add_variation(move)
        else:
            try:
                result = sf_engine.play(board, chess.engine.Limit(time=time_limit))
                board.push(result.move)
                node = node.add_variation(result.move)
            except Exception as e:
                 print(f"Stockfish Error: {e}")
                 res_str = "1-0" if nnue_color == chess.WHITE else "0-1"
                 break
    
    if visual:
        render_game(screen, bg, piece_images, board, f"Game Over: {board.result()}")
        pygame.time.delay(500)

    if res_str == "*":
        res_str = board.result()
        
    game.headers["Result"] = res_str
    
    # Update Score logic
    if res_str == "1/2-1/2":
        score = 0.5
    elif res_str == "1-0":
        if nnue_color == chess.WHITE: score = 1
    elif res_str == "0-1":
        if nnue_color == chess.BLACK: score = 1
        
    print(f"Round {round_num} vs SF {elo}: {res_str}")
    print(game, file=pgn_file_handle, end="\n\n")
    pgn_file_handle.close()
    
    sf_engine.quit()
    return score

def benchmark(model_path, stockfish_path, time_limit=0.1, visual=False, rounds=4):
    if os.path.exists(PGN_FILE):
        os.remove(PGN_FILE)

    if not os.path.exists(stockfish_path):
        # ... (check path)
        pass

    print("Loading NNUE Engine...")
    nnue = NNUEEngine(model_path)
    
    elos = [1350, 1600, 2000] 
    
    total_score = 0
    total_games = 0
    
    if visual and VISUAL_AVAILABLE:
        pygame.init()

    # Round Robin
    for r in range(1, rounds + 1):
        print(f"\n--- Round {r} ---")
        for elo in elos:
            play_single_game(nnue, stockfish_path, elo, r, time_limit=time_limit, visual=visual)
            
    if visual and VISUAL_AVAILABLE:
        pygame.quit()
            
    print(f"\nBenchmark Complete. Games saved to {PGN_FILE}")
    run_bayeselo()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/checkpoint_latest.pth")
    parser.add_argument("--stockfish", type=str, default="stockfish.exe")
    parser.add_argument("--visual", action="store_true", help="Enable live visualization")
    parser.add_argument("--time", type=float, default=2.0, help="Time limit per move in seconds")
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds (games per opponent)")
    args = parser.parse_args()
    
    if args.model == "checkpoints/checkpoint_latest.pth" and not os.path.exists(args.model):
        if os.path.exists("checkpoints_1/checkpoint_latest.pth"):
            args.model = "checkpoints_1/checkpoint_latest.pth"
            print(f"Auto-detected model at {args.model}")

    print(f"Using model: {args.model}")

    benchmark(args.model, args.stockfish, time_limit=args.time, visual=args.visual, rounds=args.rounds)
