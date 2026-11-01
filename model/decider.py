"""Fajfuzzy's decision model: the pretrained discriminator reads a question, a state and a list
of options, and one logit per <|opt|> token becomes a probability per option.

Input layout (ids from the tokenizer's special block):
    <|cls|> <|question|> question <|state|> state <|options|>
        <|opt|> name <|criteria|> criteria ... <|sep|>
The head starts from the pretraining RTD head, so fine-tuning begins with a projection that
already reads every position of the encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.electra import Config, Encoder

PAD, CLS, SEP, STATE, QUESTION, CRITERIA, OPTIONS, OPT = 0, 1, 2, 4, 5, 6, 7, 8


class Decider(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.n_embd
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.emb_ln = nn.LayerNorm(d, bias=False)
        self.disc = Encoder(d, cfg.n_layer, cfg.n_head, cfg.ffn_hidden, cfg.max_len, cfg.rope_theta)
        self.dense = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, 1)

    @classmethod
    def from_disc(cls, path, map_location="cpu"):
        ck = torch.load(str(path), map_location=map_location, weights_only=False)
        model = cls(Config(**ck["config"]))
        w = {k: v.float() for k, v in ck["weights"].items()}
        w["dense.weight"] = w.pop("rtd_dense.weight")
        w["out.weight"], w["out.bias"] = w.pop("rtd_out.weight"), w.pop("rtd_out.bias")
        model.load_state_dict(w)
        return model

    @classmethod
    def from_file(cls, path, map_location="cpu"):
        ck = torch.load(str(path), map_location=map_location, weights_only=False)
        model = cls(Config(**ck["config"]))
        model.load_state_dict({k: v.float() for k, v in ck["weights"].items()})
        return model

    def forward(self, ids, lengths):
        """ids (B, T) padded with PAD, lengths (B,). Logits (B, T), -inf everywhere except <|opt|>."""
        T = ids.shape[1]
        keep = torch.arange(T, device=ids.device)[None, :] < lengths[:, None]
        h = self.disc(self.emb_ln(self.tok(ids)), keep[:, None, None, :])
        logits = self.out(F.gelu(self.dense(h), approximate="tanh")).squeeze(-1).float()
        return logits.masked_fill(ids != OPT, float("-inf"))


def encode(tok, question, state, options, criteria=None, max_len=512, crit_cap=48, question_cap=160):
    """Token ids for one decision. Options always fit; the state takes what is left, keeping its
    first two thirds and last third when it is too long."""
    ids = lambda s: tok.encode(s).ids if s else []
    criteria = criteria or [None] * len(options)
    chars = max_len * 8
    if state and len(state) > chars:
        state = state[:chars * 2 // 3] + state[-(chars // 3):]
    tail = [OPTIONS]
    for name, crit in zip(options, criteria):
        tail += [OPT] + ids(str(name))[:64]
        if crit:
            tail += [CRITERIA] + ids(str(crit))[:crit_cap]
    tail.append(SEP)
    head = [CLS, QUESTION] + ids(question)[:question_cap] + [STATE]
    room = max_len - len(head) - len(tail)
    if room < 8:
        raise ValueError(f"{len(options)} options do not fit in {max_len} tokens")
    body = ids(state)
    if len(body) > room:
        first = room * 2 // 3
        body = body[:first] + body[len(body) - (room - first):]
    return head + body + tail


def soft_loss(logits, targets):
    """Cross-entropy against a target distribution per row. targets (B, T) holds each option's
    share at its <|opt|> position and zeros elsewhere."""
    logp = torch.log_softmax(logits, -1).masked_fill(targets == 0, 0.0)
    return -(targets * logp).sum(-1).mean()


def option_probs(logits, ids):
    """Per row, the probabilities over its <|opt|> positions, in input order."""
    return [torch.softmax(row[x == OPT], -1) for row, x in zip(logits, ids)]
