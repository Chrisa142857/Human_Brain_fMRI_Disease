import torch, math
import torch.nn as nn
from datetime import datetime

from spdnet import SPDTransform, SPDRectified, SPDTangentSpace, Normalize
from config import BASELINE_MODEL

class Baseline(nn.Module):
    def __init__(self, matrix_size=None) -> None:
        super().__init__()
        if matrix_size is not None:
            self.conv = nn.Sequential(nn.Conv2d(1, BASELINE_MODEL['embed_dim'], matrix_size, matrix_size))
        else:
            self.linear = nn.Linear(BASELINE_MODEL['in_dim'], BASELINE_MODEL['embed_dim'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=BASELINE_MODEL['embed_dim'], nhead=BASELINE_MODEL['nhead'])
        self.pos_encoder = PositionalEncoding(BASELINE_MODEL['embed_dim'])
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=BASELINE_MODEL['nlayer'])
        self.classifier = nn.Linear(BASELINE_MODEL['embed_dim'], BASELINE_MODEL['nclass'])

    def forward(self, x):
        ## x.shape [batch, timeseries, roi_num]
        if len(x.shape) > 3: 
            x = self.conv(x.transpose(1, 0)).squeeze().unsqueeze(0)
        else:
            x = self.linear(x)
        x = self.pos_encoder(x.transpose(1, 0))
        x = self.encoder(x)
        # x = x.mean(dim=0)
        x = x.max(dim=0)[0]
        x = self.classifier(x)
        return x
       
class BaselineSPD(nn.Module):
    def __init__(self, matrix_size) -> None:
        super().__init__()
        embed_dim = 2048
        self.spdnet = SPDNet(matrix_size)
        # self.triu_ind = torch.triu_indices(matrix_size, matrix_size)
        # self.spdnet.out_size = self.triu_ind.shape[1]
        self.linear1 = nn.Linear(self.spdnet.out_size, embed_dim)
        self.linears = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        ) for _ in range(20)])
        self.classifier = nn.Linear(embed_dim, BASELINE_MODEL['nclass'])

    def forward(self, x):
        ## x.shape [batch, roi_num, roi_num] (SPD matrix)
        x = self.spdnet(x)#.unsqueeze(1)
        # x = x[:, self.triu_ind[0], self.triu_ind[1]]
        x = self.linear1(x)
        for layer in self.linears:
            x = x + layer(x)
        x = self.classifier(x)
        return x
            
class BaselineSPDTransformer(nn.Module):
    def __init__(self, matrix_size) -> None:
        super().__init__()
        self.spdnet = SPDNet(matrix_size)
        self.triu_ind = torch.triu_indices(matrix_size, matrix_size)
        self.spdnet.out_size = self.triu_ind.shape[1]
        self.linear = nn.Linear(self.spdnet.out_size, BASELINE_MODEL['embed_dim'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=BASELINE_MODEL['embed_dim'], nhead=BASELINE_MODEL['nhead'])
        self.pos_encoder = PositionalEncoding(BASELINE_MODEL['embed_dim'])
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=BASELINE_MODEL['nlayer'])
        self.classifier = nn.Linear(BASELINE_MODEL['embed_dim'], BASELINE_MODEL['nclass'])

    def forward(self, x):
        ## x.shape [batch, timeseries, roi_num, roi_num] (SPD matrix)
        # print(datetime.now(), 'Start SPD Net')
        # x = self.spdnet(x[0]).unsqueeze(0)# batch, timeseries, channel
        x = x[:, :, self.triu_ind[0], self.triu_ind[1]]
        # print(datetime.now(), 'Done SPD Net')
        x = self.linear(x)
        x = self.pos_encoder(x.transpose(1, 0))
        x = self.encoder(x)
        # x = x.mean(dim=0)
        x = x.max(dim=0)[0]
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
    

class SPDNet(nn.Module):
    def __init__(self, matrix_size=150, layer_num=4):
        super(SPDNet, self).__init__()
        self.out_size = [matrix_size // (2**i) for i in range(layer_num)]
        layer_list = []
        in_size = matrix_size
        for out_size in self.out_size:
            layer_list.append(SPDTransform(in_size, out_size))
            layer_list.append(SPDRectified())
            in_size = out_size

        # layer_list.append(SPDTangentSpace(out_size, vectorize_all=False))
        layer_list.append(SPDTangentSpace(out_size))
        layer_list.append(Normalize())
        self.layers = nn.Sequential(*layer_list)
        # self.layers = layer_list
        self.out_size = out_size + int(out_size*(out_size-1)/2)

    def forward(self, x):
        x = self.layers(x)
        # for li, layer in enumerate(self.layers):
        #     print("SPDNet layer", li, layer)
        #     x = layer(x)
        return x
 