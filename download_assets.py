import os
import requests

# URL template for chessboard.js wikipedia pieces
BASE_URL = "https://raw.githubusercontent.com/oakmac/chessboardjs/master/website/img/chesspieces/wikipedia/{}.png"

ASSETS_DIR = r"e:\Chess\assets"
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)

PIECES = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']

def download_assets():
    print(f"Downloading assets to {ASSETS_DIR}...")
    for p in PIECES:
        url = BASE_URL.format(p)
        filename = os.path.join(ASSETS_DIR, f"{p}.png")
        if os.path.exists(filename):
            print(f"Skipping {p}, already exists.")
            continue
            
        print(f"Downloading {p} from {url}...")
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f"Failed to download {p}: {e}")

if __name__ == "__main__":
    download_assets()
