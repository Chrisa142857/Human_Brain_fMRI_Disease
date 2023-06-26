import torch, math
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class MultiheadCorrelation(nn.Module):
    
    def __init__(self, d_model, nhead):
        self.nhead = nhead
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
    
    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        head_dim = embed_dim // self.nhead
        assert head_dim * self.nhead == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {self.nhead}"
        assert query is key and key is value
        qkv = self.qkv_proj(query)
        # use the reshape way as PyTorch official code
        q, k, v = qkv.unflatten(-1, (3, query.size(-1))).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q = q.view(tgt_len, bsz * self.nhead, head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * self.nhead, head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * self.nhead, head_dim).transpose(0, 1)
        src_len = k.size(1)
        if not self.training:
            dropout_p = 0.0
        q = q.view(bsz, self.nhead, tgt_len, head_dim)
        k = k.view(bsz, self.nhead, src_len, head_dim)
        v = v.view(bsz, self.nhead, src_len, head_dim)
        ## self attention
        # scale_factor = 1 / math.sqrt(q.size(-1))
        # attn = q @ k.transpose(-2, -1) * scale_factor
        # attn = torch.softmax(attn, dim=-1)
        # attn = torch.dropout(attn, dropout_p)
        # return attn_weight @ v
        ## correlation coefficient
        corr = corrcoef(q, k)
        corr = torch.softmax(corr, dim=-1)
        corr = torch.dropout(corr, dropout_p)
        return corr @ v



def corrcoef(X, Y):
    X = X - X.mean(-1).unsqueeze(-1)
    Y = Y - Y.mean(-1).unsqueeze(-1)
    Y_T = Y.swapaxes(-2, -1)
    c = X @ Y_T
    d = torch.diagonal(c, 0, -2, -1)
    stddev = d.sqrt()
    c = c / stddev[..., None]
    c = c / stddev[..., None, :]
    c = torch.clip(c, -1, 1)
    c[c.isnan()] = 0
    return c
