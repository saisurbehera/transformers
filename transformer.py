import einops
from fancy_einsum import einsum
from dataclasses import dataclass
import torch as t
import torch.nn as nn
import numpy as np
import math

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(self, residual):
        # residual: [batch, position, d_model]
        # output: [batch, position, d_model]
        mean = residual.mean(dim=-1, keepdim=True)
        variance = residual.var(dim=-1, keepdim=True,correction=0) + self.cfg.layer_norm_eps
        
        residual = (residual-mean)/(variance**0.5)
        return residual*self.w  + self.b
    
class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        # output: [batch, position, d_model]
        return self.W_E[tokens]

class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        # output: [batch, position, d_model]
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)

class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.scale = cfg.d_head**0.5
        self.softmaxi = nn.Softmax(dim=-1)
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device="cuda"))

    def forward(self, normalized_resid_pre: t.Tensor):
        # normalized_resid_pre: [batch, position, d_model]
        # output: [batch, position, d_model]

        # Calculate query, key and value vectors
        ## Get the query matrix
        query_mat = einsum("batch position_q d_model, n_heads d_model d_head -> batch position_q n_heads d_head", normalized_resid_pre, self.W_Q) + self.b_Q 
        key_mat = einsum("batch position_k d_model, n_heads d_model d_head -> batch position_k n_heads d_head", normalized_resid_pre, self.W_K) + self.b_K
        val_mat = einsum("batch position_v d_model, n_heads d_model d_head -> batch position_v n_heads d_head", normalized_resid_pre, self.W_V) + self.b_V
        
        # Calculate the attention scores 
        # The first two positions of batch and head make logical sense
        # Last two blocks are essential a matrix of seq_length tp seq_length
        atten_qk = einsum("batch position_q n_heads d_head , batch position_k n_heads d_head -> batch n_heads position_q position_k ", query_mat, key_mat) 
        atten = (self.apply_causal_mask(atten_qk/self.scale)).softmax(-1)
        
        val_mat_res = einsum("batch position_v n_heads d_head , batch n_heads position_q position_v -> batch position_q n_heads d_head",  val_mat,atten)
        
        attn_out = einsum("batch position_q n_heads d_head,  n_heads d_head d_model -> batch position_q d_model", val_mat_res, self.W_O) + self.b_O

        return attn_out

    def apply_causal_mask(self, attn_scores: t.Tensor):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        # output: [batch, n_heads, query_pos, key_pos]

        # This code convert the top part of the matric into
        mask = t.triu(t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        # Apply the mask to attention scores, then return the masked scores
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores
    
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))

    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        # output: [batch, position, d_model]
        ll1 = einsum("batch position d_model , d_model d_mlp -> batch position d_mlp",normalized_resid_mid,self.W_in) + self.b_in
        act1 = gelu_new(ll1)
        ll2 = einsum("batch position d_mlp , d_mlp d_model -> batch position d_model",act1,self.W_out) + self.b_out
        return ll2
        
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre):
        # resid_pre [batch, position, d_model]
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(self, normalized_resid_final):
        # normalized_resid_final: [batch, position, d_model]
        # output: [batch, position, d_vocab]
        return einsum("batch position d_model,  d_model d_vocab ->  batch position d_vocab",normalized_resid_final,self.W_U ) + self.b_U 

class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        # tokens [batch, position]
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits