import torch
import torch.nn as nn
from config import BASELINE_MODEL


class Baseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(BASELINE_MODEL['in_dim'], BASELINE_MODEL['embed_dim'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=BASELINE_MODEL['embed_dim'], nhead=BASELINE_MODEL['nhead'])
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=BASELINE_MODEL['nlayer'])
        self.classifier = nn.Linear(BASELINE_MODEL['embed_dim'], BASELINE_MODEL['nclass'])

    def forward(self, x):
        x = self.linear(x)
        x = self.encoder(x)
        x = self.classifier(x)
        return x
        