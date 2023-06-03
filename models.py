import torch, math
import torch.nn as nn
from config import BASELINE_MODEL


class Baseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(BASELINE_MODEL['in_dim'], BASELINE_MODEL['embed_dim'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=BASELINE_MODEL['embed_dim'], nhead=BASELINE_MODEL['nhead'])
        self.pos_encoder = PositionalEncoding(BASELINE_MODEL['embed_dim'])
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=BASELINE_MODEL['nlayer'])
        self.classifier = nn.Linear(BASELINE_MODEL['embed_dim'], BASELINE_MODEL['nclass'])

    def forward(self, x):
        ## x.shape [batch, timeseries, roi_num]
        x = self.linear(x)
        x = self.pos_encoder(x.transpose(1, 0))
        x = self.encoder(x)
        x = x.mean(dim=0)
        x = self.classifier(x)
        return x
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)