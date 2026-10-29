"""G-Decide's encoder, trained from scratch with ELECTRA: a small generator fills masked
tokens, the discriminator spots which tokens were replaced.

Pre-norm blocks with RoPE, GeGLU and no biases. The generator is a third of the
discriminator's width and shares the token table the DeBERTa-v3 way: the generator
trains the table, the discriminator reads a stop-gradient copy plus its own delta.
After pretraining only the discriminator is kept (Encoder plus merged embeddings).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    ffn_hidden: int = 2048
    gen_layer: int = 12
    gen_head: int = 4
    gen_embd: int = 256
    gen_ffn: int = 704
    max_len: int = 2048
    rope_theta: float = 50000.0
    mask_rate: float = 0.15
    rtd_weight: float = 50.0
    n_special: int = 32
    pad_id: int = 0
    cls_id: int = 1
    sep_id: int = 2
    mask_id: int = 3


TINY = dict(n_layer=2, n_head=2, n_embd=64, ffn_hidden=128,
            gen_layer=2, gen_head=2, gen_embd=32, gen_ffn=64)


def rope_tables(head_dim, max_len, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len).float(), inv)
    return freqs.cos(), freqs.sin()


def rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class Block(nn.Module):
    def __init__(self, d, n_head, ffn):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d, bias=False)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d, bias=False)
        self.up = nn.Linear(d, 2 * ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, x, cos, sin, mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.n_head, D // self.n_head).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q, cos, sin), rope(k, cos, sin), v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        gate, up = self.up(self.ln2(x)).chunk(2, dim=-1)
        return x + self.down(F.gelu(gate, approximate="tanh") * up)


class Encoder(nn.Module):
    def __init__(self, d, n_layer, n_head, ffn, max_len, theta):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, n_head, ffn) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d, bias=False)
        cos, sin = rope_tables(d // n_head, max_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, mask=None):
        T = x.shape[1]
        assert T <= self.cos.shape[0], f"sequence of {T} tokens exceeds max_len {self.cos.shape[0]}"
        cos, sin = self.cos[:T], self.sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        return self.ln_f(x)


def init_weights(module, n_layers):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
    for m in module.modules():
        if isinstance(m, Encoder):
            for block in m.blocks:
                for w in (block.proj.weight, block.down.weight):
                    nn.init.normal_(w, std=0.02 / math.sqrt(2 * n_layers[m]))


class Electra(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, g = cfg.n_embd, cfg.gen_embd
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.delta = nn.Embedding(cfg.vocab_size, d)
        self.emb_ln = nn.LayerNorm(d, bias=False)
        self.disc = Encoder(d, cfg.n_layer, cfg.n_head, cfg.ffn_hidden, cfg.max_len, cfg.rope_theta)
        self.rtd_dense = nn.Linear(d, d, bias=False)
        self.rtd_out = nn.Linear(d, 1)
        self.gen_emb_ln = nn.LayerNorm(d, bias=False)
        self.gen_in = nn.Linear(d, g, bias=False)
        self.gen = Encoder(g, cfg.gen_layer, cfg.gen_head, cfg.gen_ffn, cfg.max_len, cfg.rope_theta)
        self.gen_dense = nn.Linear(g, d, bias=False)
        self.gen_ln = nn.LayerNorm(d, bias=False)
        self.gen_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        init_weights(self, {self.disc: cfg.n_layer, self.gen: cfg.gen_layer})
        nn.init.zeros_(self.delta.weight)

    def disc_params(self):
        parts = (self.emb_ln, self.disc, self.rtd_dense, self.rtd_out)
        return self.tok.weight.numel() + sum(p.numel() for m in parts for p in m.parameters())

    def gen_params(self):
        parts = (self.gen_emb_ln, self.gen_in, self.gen, self.gen_dense, self.gen_ln)
        return sum(p.numel() for m in parts for p in m.parameters()) + self.gen_bias.numel()

    def forward(self, ids):
        cfg = self.cfg
        sel = (torch.rand(ids.shape, device=ids.device) < cfg.mask_rate) & (ids >= cfg.n_special)
        masked = ids.masked_fill(sel, cfg.mask_id)
        hidden = self.gen(self.gen_in(self.gen_emb_ln(self.tok(masked))))[sel]
        head = self.gen_ln(F.gelu(self.gen_dense(hidden), approximate="tanh"))
        logits = F.linear(head, self.tok.weight, self.gen_bias).float()
        target = ids[sel]
        mlm = F.cross_entropy(logits, target)

        with torch.no_grad():
            u = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-7)
            sampled = (logits - torch.log(-torch.log(u))).argmax(-1)
            corrupted = ids.masked_scatter(sel, sampled)
            replaced = (corrupted != ids).float()

        emb = F.embedding(corrupted, self.tok.weight.detach()) + self.delta(corrupted)
        h = self.disc(self.emb_ln(emb))
        rtd_logits = self.rtd_out(F.gelu(self.rtd_dense(h), approximate="tanh")).squeeze(-1).float()
        rtd = F.binary_cross_entropy_with_logits(rtd_logits, replaced)
        loss = mlm + cfg.rtd_weight * rtd

        with torch.no_grad():
            rtd_acc = ((rtd_logits > 0).float() == replaced).float().mean()
            gen_acc = (logits.argmax(-1) == target).float().mean()
            replaced_rate = replaced.sum() / sel.sum().clamp(min=1)
        return torch.stack([loss, mlm.detach(), rtd.detach(), rtd_acc, gen_acc, replaced_rate]).unsqueeze(0)


STATS = ("loss", "mlm", "rtd", "rtd_acc", "gen_acc", "replaced")


def discriminator_state(model: Electra):
    sd = model.state_dict()
    out = {"tok.weight": sd["tok.weight"] + sd["delta.weight"]}
    for k, v in sd.items():
        if k.startswith(("emb_ln.", "disc.", "rtd_dense.", "rtd_out.")):
            out[k] = v
    return out
