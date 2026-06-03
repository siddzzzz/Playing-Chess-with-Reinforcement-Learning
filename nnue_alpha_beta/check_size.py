import torch
import os

def check(path):
    if not os.path.exists(path):
        print(f"{path}: Not Found")
        return
        
    try:
        c = torch.load(path, map_location='cpu')
        
        # Check if state dict is nested
        if 'model_state_dict' in c:
            sd = c['model_state_dict']
            epoch = c.get('epoch', '?')
        else:
            sd = c
            epoch = '?'
            
        # Get shape of first hidden layer weight
        shape = sd['l1.weight'].shape
        print(f"File: {path}")
        print(f"  - Epoch: {epoch}")
        print(f"  - Hidden Layer Shape: {shape}")
        print(f"  - Hidden Size: {shape[0]} (Input {shape[1]})")
        print("-" * 30)
        
    except Exception as e:
        print(f"{path}: Error {e}")

print("--- Analysis ---")
check('checkpoints/checkpoint_epoch_100.pth')
check('checkpoints_1/checkpoint_latest.pth')
check('checkpoints/checkpoint_latest.pth')
