import json
import os

def read_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return f.read()
    return f"# {filename} not found"

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True)
    }

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(True)
    }

def main():
    cells = []
    cells.append(create_markdown_cell("# ChessGPT: Transformer-based AlphaZero\nThis notebook trains a decision transformer on chess moves."))
    cells.append(create_code_cell("!pip install python-chess torch numpy tqdm"))
    cells.append(create_code_cell("import torch\nprint(torch.cuda.is_available())"))
    
    cells.append(create_markdown_cell("## 1. Utils"))
    cells.append(create_code_cell(f"%%writefile utils.py\n{read_file('utils.py')}"))
    
    cells.append(create_markdown_cell("## 2. Transformer Model"))
    cells.append(create_code_cell(f"%%writefile model.py\n{read_file('model.py')}"))
    
    cells.append(create_markdown_cell("## 3. MCTS"))
    cells.append(create_code_cell(f"%%writefile mcts.py\n{read_file('mcts.py')}"))
    
    cells.append(create_markdown_cell("## 4. Trainer"))
    cells.append(create_code_cell(f"%%writefile trainer.py\n{read_file('trainer.py')}"))
    
    cells.append(create_markdown_cell("## 5. Run"))
    cells.append(create_code_cell("from trainer import Trainer\nt = Trainer()\nt.train(20)"))
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open("ChessGPT.ipynb", "w") as f:
        json.dump(notebook, f, indent=1)
        
    print("Notebook ChessGPT.ipynb created.")

if __name__ == "__main__":
    main()
