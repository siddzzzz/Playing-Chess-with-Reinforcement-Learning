import os
import subprocess
import argparse
import sys
import random

# Configuration
STOCKFISH_PATH = "stockfish.exe" # User must provide absolute path if not in CWD
MY_ENGINE_CMD = f"python uci.py --model checkpoints/checkpoint_latest.pth"
CUTECHESS_CLI = r"C:\Users\Siddharths\AppData\Local\Programs\Cute Chess\cutechess-cli.exe"
BAYESELO = "bayeselo"

OPENINGS_FILE = "openings.pgn" # Must exist

def check_tools():
    # Check for Stockfish
    if not os.path.exists(STOCKFISH_PATH):
        print(f"[ERROR] Stockfish not found at {STOCKFISH_PATH}")
        return False
        
    # Check for cutechess
    if os.path.exists(CUTECHESS_CLI):
        pass # Local file exists
    else:
        # Check PATH
        try:
            subprocess.run([CUTECHESS_CLI, "--version"], capture_output=True, check=False)
        except FileNotFoundError:
            print("[ERROR] cutechess-cli.exe not found in folder or PATH!")
            return False

    # Check for bayeselo (optional but requested)
    try:
        subprocess.run([BAYESELO], input="quit\n", encoding="utf-8", capture_output=True, check=False)
    except FileNotFoundError:
        print("[WARNING] bayeselo not found. Rating calculation will be skipped.")
        
    return True

def generate_openings_if_missing():
    if not os.path.exists(OPENINGS_FILE):
        print(f"Generating simple {OPENINGS_FILE}...")
        # Create a few simple start positions
        with open(OPENINGS_FILE, "w") as f:
            f.write('[Event "?"]\n[Site "?"]\n[Date "????.??.??"]\n[Round "?"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n1. e4 e5 *\n\n')
            f.write('[Event "?"]\n[Site "?"]\n[Date "????.??.??"]\n[Round "?"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n1. d4 d5 *\n\n')
            f.write('[Event "?"]\n[Site "?"]\n[Date "????.??.??"]\n[Round "?"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n1. Nf3 d5 *\n\n')

def run_tournament(rounds=2, tc="5+0.1"): # Start with 5s + 0.1s inc
    if not check_tools():
        return

    generate_openings_if_missing()

    # Define Opponents (Stockfish calibrated levels)
    # Stockfish 17+ requires UCI_Elo >= 1320.
    # We will use valid ELOs. For <1320, we would need Skill Level, but let's stick to valid Elo range for simpler config.
    opponents = [1350, 1600, 2000]
    
    # Construct Command
    cmd = [CUTECHESS_CLI]
    cmd += ["-tournament", "gauntlet"] 
    cmd += ["-concurrency", "1"] 
    cmd += ["-pgnout", "benchmark.pgn"]
    cmd += ["-each", f"tc={tc}", "proto=uci"]
    cmd += ["-openings", f"file={OPENINGS_FILE}", "order=random"]
    cmd += ["-rounds", str(rounds)]
    cmd += ["-games", "2"] 
    
    # Hero: Our Engine
    # Must run as a command
    # Use absolute path for safety
    uci_script = os.path.abspath("uci.py")
    # DETECT CORRECT CHECKPOINT (User switched to checkpoints_1)
    if os.path.exists("checkpoints_1/checkpoint_latest.pth"):
        model_path = os.path.abspath("checkpoints_1/checkpoint_latest.pth")
    else:
        model_path = os.path.abspath("checkpoints/checkpoint_latest.pth")
        
    cmd += ["-engine", f"name=MyNNUE", f"cmd={sys.executable}", "arg=-u", f"arg={uci_script}", f"arg=--model", f"arg={model_path}"]
    
    # Enable debug to see engine crashes
    # cmd += ["-debug"] 
    
    # Opponents
    for elo in opponents:
        cmd += ["-engine", f"name=Stockfish_{elo}", f"cmd={STOCKFISH_PATH}", 
                f"option.UCI_LimitStrength=true", f"option.UCI_Elo={elo}"]
                
    print("\nStarting Tournament...")
    print("Command:", " ".join(cmd))
    
    if "cutechess.exe" in cmd[0]:
        print("\n[INFO] Launching CuteChess GUI. The terminal will pause until you close the GUI window.")
        print("[INFO] Please check your taskbar if the window does not appear immediately.")
    
    try:
        # Stream output directly to console so user sees progress
        subprocess.run(cmd, check=True)
        print("\nTournament Completed. Games saved to benchmark.pgn")
        analyze_results()
    except subprocess.CalledProcessError as e:
        print(f"\nTournament Failed (Exit {e.returncode}):")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
    except Exception as e:
         print(f"Error: {e}")

def analyze_results():
    if subprocess.run([BAYESELO], input="quit\n", encoding="utf-8", capture_output=True, check=False).returncode != 0:
        return

    script = """
readpgn benchmark.pgn
elo
mm
ratings
x
quit
"""
    with open("bayes.script", "w") as f:
        f.write(script)
        
    print("\nRunning BayesElo Analysis...")
    try:
        # Run BayesElo with input redirection
        with open("bayes.script", "r") as f:
            # We use stdin=f to pipe the script file
            result = subprocess.run([BAYESELO], stdin=f, capture_output=True, text=True, check=True)
            
        print("\n--- BayesElo Ratings ---")
        print(result.stdout)
    except Exception as e:
        print(f"BayesElo Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds per opponent")
    parser.add_argument("--tc", type=str, default="5+0.1", help="Time Control (e.g., 40/60 = 40 moves in 60s, or 5+0.1 = 5s + 0.1s inc)")
    parser.add_argument("--gui", action="store_true", help="Run in CuteChess GUI to watch games")
    args = parser.parse_args()
    
    # Swap to GUI executable if requested
    if args.gui:
        CUTECHESS_CLI = CUTECHESS_CLI.replace("cutechess-cli.exe", "cutechess.exe")
        print(f"Switching to GUI: {CUTECHESS_CLI}")
    
    run_tournament(rounds=args.rounds, tc=args.tc)
