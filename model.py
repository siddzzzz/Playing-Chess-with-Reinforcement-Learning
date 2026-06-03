import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=4, num_channels=64):
        super(ChessNet, self).__init__()
        # Input: 12 channels (6 white pieces, 6 black pieces)
        self.start_conv = nn.Conv2d(12, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_start = nn.BatchNorm2d(num_channels)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        # Output: 4096 (64*64 moves)
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)
        
        # Value Head
        # Output: 1 scalar (tanh)
        self.value_conv = nn.Conv2d(num_channels, 3, kernel_size=1) # 3 channels roughly
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: [batch, 12, 8, 8]
        out = F.relu(self.bn_start(self.start_conv(x)))
        
        for block in self.res_blocks:
            out = block(out)
            
        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(1) # batch, 32*8*8
        p = self.policy_fc(p)
        # Note: We return raw logits (log_softmax is done in loss or implicitly)
        # But MCTS usually expects probabilities. We'll softmax outside or here.
        # Returning logits for numerical stability in training logic usually.
        # But for inference, we want softmax. 
        # Let's return logits for policy, values for value.
        
        # Value
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v
