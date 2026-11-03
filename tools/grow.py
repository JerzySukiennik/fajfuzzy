"""Grow Fajfuzzy v1's 12-layer discriminator into a 24-layer one that computes exactly the same.

Every pretrained block gets a copy inserted right above it, LLaMA-Pro style: the copy starts
with zero output projections (proj, down), so it adds nothing to the residual stream until
training moves it, and the grown model starts where v1 ended. Generator, embeddings and the
RTD head stay as they are. The result is an init file for train/pretrain.py --init (weights
and config, no optimiser state). The check runs both models on the same random batch.

    python tools/grow.py --ckpt s6/ckpt.pt --out grow_init.pt
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.electra import Config, Electra  # noqa: E402

ZEROED = ("proj.weight", "down.weight")


def grow(sd, cfg):
    new = Config(**{**asdict(cfg), "n_layer": 2 * cfg.n_layer})
    out = {}
    for k, v in sd.items():
        if not k.startswith("disc.blocks."):
            out[k] = v
            continue
        i, rest = k[len("disc.blocks."):].split(".", 1)
        out[f"disc.blocks.{2 * int(i)}.{rest}"] = v
        out[f"disc.blocks.{2 * int(i) + 1}.{rest}"] = torch.zeros_like(v) if rest in ZEROED else v.clone()
    return out, new


@torch.no_grad()
def same_output(old_sd, old_cfg, new_sd, new_cfg, n=2, block=128):
    a, b = Electra(old_cfg).eval(), Electra(new_cfg).eval()
    a.load_state_dict(old_sd)
    b.load_state_dict(new_sd)
    ids = torch.randint(old_cfg.n_special, old_cfg.vocab_size, (n, block))
    ids[:, 0] = old_cfg.cls_id
    torch.manual_seed(0)
    ra = a(ids)
    torch.manual_seed(0)
    rb = b(ids)
    return float((ra - rb).abs().max()), ra[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="full pretraining ckpt.pt (with the generator)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["config"])
    sd, new = grow(ck["model"], cfg)
    diff, stats = same_output(ck["model"], cfg, sd, new)
    print(f"v1 step {ck['step']}, {ck.get('tokens', 0) / 1e9:.2f}B tokens: {cfg.n_layer} -> {new.n_layer} layers, "
          f"max output difference {diff:.2e} (loss, mlm, rtd, rtd_acc, gen_acc, replaced = {[round(x, 4) for x in stats]})")
    if diff > 1e-4:
        raise SystemExit("grown model does not reproduce v1")
    torch.save({"model": sd, "config": asdict(new), "grown_from": {"step": ck["step"], "tokens": ck.get("tokens", 0)}}, a.out)
    print(f"saved {a.out} ({a.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
