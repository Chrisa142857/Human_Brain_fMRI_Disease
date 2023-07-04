import torch, math
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple

from datasets import nearestPD

class MultiheadCorrelation(nn.Module):
    
    def __init__(self, d_model, nhead, dropout=0.1, batch_first=False):
        super().__init__()
        self.nhead = nhead
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.qkv_proj_T = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.corr_loss = None
    
    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = False,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        corr = None
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        head_dim = embed_dim // self.nhead
        assert head_dim * self.nhead == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {self.nhead}"
        assert query is key and key is value
        if not self.training:
            dropout_p = 0.0
        qkv = self.qkv_proj(query)
        q, k, v = unflatten_qkv(qkv, query.size(-1), tgt_len, bsz, self.nhead, head_dim)
        # corr = corrcoef(q.transpose(-2, -1), k.transpose(-2, -1))
        # corr = corrcoef(q.transpose(-2, -1), k.transpose(-2, -1))
        # self.corr_loss = (corr - corr.transpose(-2, -1)).abs().tanh().mean([0,2,3])
        # qkv_T = self.qkv_proj_T(query)
        # q_T, k_T, v_T = unflatten_qkv(qkv_T, query.size(-1), tgt_len, bsz, self.nhead, head_dim)
        # q_T = q_T.transpose(-2, -1)
        # k_T = k_T.transpose(-2, -1)
        # v_T = v_T.transpose(-2, -1)
        ## correlation coefficient
        corr = corrcoef(q.transpose(-2, -1), k.transpose(-2, -1))
        corr = torch.softmax(corr, dim=-2)
        corr = torch.dropout(corr, self.dropout_p, train=self.training)
        assert not corr.isnan().any()
        attn_output = v @ corr
        ## corr loss
        # corr_q = self_corrcoef(q.transpose(-2, -1))
        # corr_k = self_corrcoef(k.transpose(-2, -1))
        # corr_v = self_corrcoef(v.transpose(-2, -1))
        # corr = (corr_q + corr_k + corr_v) / 3
        # corr_std = (corr_q - corr_k) + (corr_k - corr_v)
        # corr_std = corr_std.abs().tanh()
        # self.corr_loss = corr_std.mean([0,2,3])
        # ## self attention
        # attn_output = self_attn(q, k, v, self.dropout_p, self.training)
        # ## self attention T
        # q_T = q.transpose(-2, -1)
        # k_T = k.transpose(-2, -1)
        # v_T = v.transpose(-2, -1)
        # attn_output_T = self_attn(q_T, k_T, v_T, self.dropout_p, self.training)
        # attn_output = torch.cat([attn_output, attn_output_T.transpose(-2, -1)], dim=-1)
        ##################
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        return attn_output, corr

def unflatten_qkv(qkv, ch, tgt_len, bsz, nhead, head_dim):
    qkv = qkv.unflatten(-1, (3, ch)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q.view(tgt_len, bsz * nhead, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * nhead, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * nhead, head_dim).transpose(0, 1)
    src_len = k.size(1)
    q = q.view(bsz, nhead, tgt_len, head_dim)
    k = k.view(bsz, nhead, src_len, head_dim)
    v = v.view(bsz, nhead, src_len, head_dim)
    return q, k, v

def self_attn(q, k, v, dropout_p, training):
    scale_factor = 1 / math.sqrt(q.size(-1))
    attn = q @ k.transpose(-2, -1) * scale_factor
    attn = torch.softmax(attn, dim=-1)
    attn = torch.dropout(attn, dropout_p, train=training)
    attn_output = attn @ v
    return attn_output

def self_corrcoef(X):
    X = X - X.mean(-1,keepdim=True)
    Y_T = X.transpose(-2, -1)
    c = X @ Y_T # sum{(xi-x)(yi-y)}
    assert not c.isnan().any()
    stddev = (X**2).sum(-1,keepdim=True) @ (Y_T**2).sum(-2,keepdim=True)
    assert not stddev.isnan().any()
    stddev = stddev.sqrt() # sqrt{sum((xi-x)^2)sum((yi-y)^2)}
    stddev = (X**2).sum(-1,keepdim=True) @ (Y_T**2).sum(-2,keepdim=True)
    assert not stddev.isnan().any()
    assert c.shape == stddev.shape
    c = c / stddev
    assert not c.isnan().any()
    c = torch.clip(c, -1, 1)
    return c


def corrcoef(X, Y):
    X = X - X.mean(-1).unsqueeze(-1)
    Y = Y - Y.mean(-1).unsqueeze(-1)
    Y_T = Y.transpose(-2, -1)
    c = X @ Y_T # sum{(xi-x)(yi-y)}
    stddev = (X**2).sum(-1,keepdim=True) @ (Y_T**2).sum(-2,keepdim=True)
    stddev = stddev.sqrt() # sqrt{sum((xi-x)^2)sum((yi-y)^2)}
    assert c.shape == stddev.shape
    c = c / stddev
    # d = torch.diagonal(c, 0, -2, -1)
    # assert not (d<0).any()
    # stddev = d.sqrt()
    # assert not stddev.isnan().any()
    # c = c / stddev[..., None]
    # c = c / stddev[..., None, :]
    c = torch.clip(c, 0, 1)
    return c
