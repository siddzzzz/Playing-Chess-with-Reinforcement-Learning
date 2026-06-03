# Playing Chess with Reinforcement Learning

A comprehensive, multi-architecture chess engine repository exploring state-of-the-art Deep Learning and Reinforcement Learning approaches to playing chess. This project includes implementations of **AlphaZero-style Policy/Value Networks**, **NNUE-based Alpha-Beta Engines**, and **Transformer-based (GPT) Chess Bots**, complete with interactive Pygame GUIs, UCI support, and Elo benchmark tools.

---

## 🌟 Architectures & Key Features

This repository is split into three main implementations:

| Feature / Architecture | 🤖 AlphaZero / Root | 🧠 NNUE + Alpha-Beta | 🚀 Transformer / GPT |
| :--- | :--- | :--- | :--- |
| **Model Architecture** | ResNet with Policy & Value heads | Feedforward ChessNN (768 -> 512 -> 32 -> 1) | `ChessTransformer` (Embedding + Attention) |
| **Search Algorithm** | Monte Carlo Tree Search (MCTS) with PUCT | Alpha-Beta Pruning with Transposition Tables | MCTS / Policy Rollouts |
| **Interface** | CLI (`play.py`) & Pygame GUI (`gui.py`) | UCI Engine (`uci.py`) & Pygame GUIs | Pygame CLI/Spectator (`spectator.py`) |
| **Evaluation Type** | Win/loss/draw value network output | Centipawn evaluation via NNUE | Board sequence token embedding attention |
| **Training Source** | Self-play reinforcement learning | Stockfish generated datasets & custom search data | Game-to-game learning with CSV export |

---

## 📂 Project Directory Structure

```directory
├── Chess_AI.ipynb             # Jupyter Notebook for Root AlphaZero training/testing
├── best_model.pth             # Pre-trained checkpoint for the root AlphaZero agent
├── model.py                   # PyTorch implementation of ChessNet (Deep Residual Network)
├── mcts.py                    # Monte Carlo Tree Search with PUCT formula selection
├── trainer.py                 # Self-play training pipeline for the AlphaZero agent
├── play.py                    # CLI loop to play against the AlphaZero agent
├── gui.py                     # Interactive Pygame GUI for Human (White) vs AI (Black)
├── utils.py                   # Helper functions (board tensor representation, move encodings)
├── download_assets.py         # Utility to download chess piece sprite images
│
├── 🧠 nnue_alpha_beta/        # Efficiently Updatable Neural Network & Alpha-Beta Engine
│   ├── model.py               # Feedforward NNUE network (sparse 768-feature input)
│   ├── engine.py              # Alpha-Beta pruning engine with transposition tables & move ordering
│   ├── train.py               # Train script for NNUE model
│   ├── data_gen.py            # Self-play dataset generator for NNUE training
│   ├── uci.py                 # UCI interface support for integration into standard GUIs
│   ├── gui_play.py            # Pygame GUI for human vs NNUE bot
│   ├── gui_versus.py          # Pygame GUI for watching engine matches
│   ├── gui_watch.py           # GUI to spectate self-play or specific matches
│   ├── benchmark_elo.py       # ELO estimation benchmarker using Cutechess and Bayeselo
│   └── Chess_Training.ipynb   # Jupyter Notebook training guide for NNUE
│
└── 🚀 v2_transformer/         # Transformer/GPT-based Chess Agent
    ├── model.py               # ChessTransformer model mapping board inputs via self-attention
    ├── trainer.py             # Policy training pipeline for ChessTransformer
    ├── mcts.py                # MCTS search modified for Transformer evaluations
    ├── play.py                # Play module for Transformer agent
    ├── spectator.py           # Match spectator loop for Transformer matches
    └── ChessGPT.ipynb         # Jupyter Notebook training guide for ChessGPT
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Playing-Chess-with-Reinforcement-Learning.git
   cd Playing-Chess-with-Reinforcement-Learning
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install required packages using:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Packages include `torch`, `numpy`, `python-chess`, `tqdm`, `pygame`, and `requests`.*

3. **Download Graphic Assets**:
   Run the following script to download standard Wikipedia chess piece sprites into the `assets` folder for the Pygame interfaces:
   ```bash
   python download_assets.py
   ```

---

## 🎮 How to Run

### 1. AlphaZero Agent (Root)
* **Play in GUI (Human vs AI)**:
  Runs the pygame window. By default, you play White and the AI plays Black.
  ```bash
  python gui.py
  ```
* **Play in CLI**:
  ```bash
  python play.py
  ```
* **Train from Scratch**:
  ```bash
  python trainer.py
  ```

### 2. NNUE Engine with Alpha-Beta
Change directory into `nnue_alpha_beta/`:
* **Play against NNUE Agent in GUI**:
  ```bash
  python gui_play.py --model checkpoints_1/checkpoint_latest.pth --time 2.0
  ```
* **Watch NNUE Engine Play vs Random**:
  ```bash
  python gui_versus.py
  ```
* **Train NNUE Network**:
  1. Generate training data: `python data_gen.py`
  2. Start training: `python train.py`
* **Run Elo Benchmarking**:
  Measures ELO ratings using Cutechess and Bayeselo:
  ```bash
  python benchmark_elo.py
  ```

### 3. Transformer Agent (ChessGPT)
Change directory into `v2_transformer/`:
* **Play against ChessGPT**:
  ```bash
  python play.py
  ```
* **Watch Transformer Play in Spectator Mode**:
  ```bash
  python spectator.py
  ```
* **Train ChessGPT**:
  ```bash
  python trainer.py
  ```

---

## 📘 Deep Dive: Architecture Details

### AlphaZero (MCTS + ChessNet)
* **State Representation**: The board state is converted into a $12 \times 8 \times 8$ tensor (6 piece types for White, 6 for Black).
* **Policy Head**: Predicts a probability distribution over all possible moves ($64 \times 64 = 4096$ output values).
* **Value Head**: Evaluates the position, outputting a scalar between $-1$ (loss) and $+1$ (win).
* **PUCT Search**: Monte Carlo Tree Search selects moves by balancing exploitation (high value) and exploration (high prior probability) using the PUCT formula:
  $$U(s, a) = c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$

### NNUE (Efficiently Updatable Neural Network)
* **Input Layer**: Features a sparse 768-element bitboard mapping piece types to specific squares.
* **Alpha-Beta Engine**: Evaluates leaf nodes using the neural network rather than static piece-square tables, allowing for deep positional awareness combined with fast search speeds.
* **Transposition Tables**: Stores previously searched board states to eliminate redundant branch evaluations.

### ChessGPT (ChessTransformer)
* **Board Sequence Embedding**: Embeds the 64 squares of the board as a sequential sequence, adding learned positional encodings to maintain file/rank references.
* **Multi-Head Attention**: Captures long-range spatial relationships on the board across multiple attention layers.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
