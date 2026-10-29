"""Pretrain Fajfuzzy's ELECTRA encoder, resumable across Kaggle sessions.

Every session restores weights, optimiser, loss scaler, step and RNG, and keeps the
schedule frozen by session 1 (steps, warmup, learning rate, batch shape, context
switch), so a chain of sessions trains like one uninterrupted run. The last part of
the run switches from 512 to 2048 tokens of context at the same tokens per step.

    python train/pretrain.py --data /kaggle/input/.../decide --out run1 --max-hours 10.5
    python train/pretrain.py --data ... --out run1 --resume --max-hours 10.5
    python train/pretrain.py --random-data --probe --max-steps 8 --batch-size 64 --grad-accum 4
"""

import argparse
import contextlib
import json
import math
import os
import queue
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.electra import STATS, TINY, Config, Electra, discriminator_state  # noqa: E402

SCHEDULE = ("max_steps", "warmup", "lr", "min_lr", "weight_decay", "batch_size", "grad_accum",
            "block", "long_block", "long_from", "long_batch_size", "long_grad_accum")


class Windows:
    def __init__(self, path, block, cfg, prefetch=4):
        self.data = np.memmap(path, dtype=np.uint16, mode="r") if path else None
        if self.data is not None and len(self.data) <= block:
            raise ValueError(f"{path} holds only {len(self.data)} tokens")
        self.block, self.cfg = block, cfg
        self.queue = queue.Queue(maxsize=prefetch)
        self.size = None

    def make(self, size):
        if self.data is None:
            x = np.random.randint(self.cfg.n_special, self.cfg.vocab_size, size=(size, self.block))
        else:
            starts = np.random.randint(0, len(self.data) - self.block, size=size)
            x = self.data[starts[:, None] + np.arange(self.block)[None, :]]
        x = np.ascontiguousarray(x, dtype=np.int64)
        x[:, 0] = self.cfg.cls_id
        return torch.from_numpy(x)

    def fill(self):
        while True:
            self.queue.put(self.make(self.size))

    def batch(self, size, device):
        if self.size is None:
            self.size = size
            threading.Thread(target=self.fill, daemon=True).start()
        assert size == self.size, "batch size changed after prefetching started"
        x = self.queue.get()
        return x.pin_memory().to(device, non_blocking=True) if device == "cuda" else x


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    r = min(1.0, (step - a.warmup) / max(1, a.max_steps - a.warmup))
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * r))


def make_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def plain_args(args):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def save(path, model, opt, scaler, step, tokens, cfg, args):
    core = model.module if hasattr(model, "module") else model
    tmp = path.with_suffix(".tmp")
    torch.save({"model": core.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "tokens": tokens, "config": asdict(cfg), "args": plain_args(args),
                "torch_rng": torch.get_rng_state()}, tmp)
    os.replace(tmp, path)


def export(path, model, cfg, step, tokens):
    core = model.module if hasattr(model, "module") else model
    weights = {k: v.half() for k, v in discriminator_state(core).items()}
    tmp = path.with_suffix(".tmp")
    torch.save({"weights": weights, "config": asdict(cfg), "step": step, "tokens": tokens}, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, data, size, iters, device, ctx):
    model.eval()
    total = torch.zeros(len(STATS))
    for _ in range(iters):
        with ctx:
            total += model(data.batch(size, device)).float().mean(0).cpu()
    model.train()
    return (total / iters).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="prefix of <data>_train.bin and <data>_val.bin")
    ap.add_argument("--random-data", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("checkpoints/run1"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=36000)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--long-block", type=int, default=2048)
    ap.add_argument("--long-from", type=float, default=0.92)
    ap.add_argument("--long-batch-size", type=int, default=16)
    ap.add_argument("--long-grad-accum", type=int, default=4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--single-gpu", action="store_true")
    args = ap.parse_args()

    ckpt_path = args.out / "ckpt.pt"
    ck = None
    if args.resume:
        if not ckpt_path.exists():
            raise SystemExit(f"--resume given but {ckpt_path} does not exist")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for k in SCHEDULE:
            if k in ck["args"]:
                setattr(args, k, ck["args"][k])
        cfg = Config(**ck["config"])
    else:
        cfg = Config(**TINY) if args.tiny else Config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda" and torch.cuda.get_device_capability()[0] < 8
    ctx = (torch.autocast(device_type="cuda", dtype=torch.float16 if fp16 else torch.bfloat16)
           if device == "cuda" else contextlib.nullcontext())
    scaler = make_scaler(fp16)

    model = Electra(cfg).to(device)
    print(f"device={device} amp={'fp16' if fp16 else 'bf16' if device == 'cuda' else 'off'}  "
          f"discriminator {model.disc_params()/1e6:.1f}M  generator {model.gen_params()/1e6:.1f}M", flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": other, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), eps=1e-6, fused=device == "cuda")

    step, tokens = 0, 0
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["torch_rng"].cpu().to(torch.uint8))
        step, tokens = ck["step"], ck.get("tokens", 0)
        print(f"resumed at step {step}, {tokens/1e9:.3f}B tokens seen", flush=True)
        del ck

    n_gpu = torch.cuda.device_count() if device == "cuda" and not args.single_gpu else 1
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    print(f"GPUs used: {n_gpu}", flush=True)

    train_path = None if args.random_data else f"{args.data}_train.bin"
    val_path = None if args.random_data else f"{args.data}_val.bin"
    short = Windows(train_path, args.block, cfg)
    long = Windows(train_path, args.long_block, cfg)
    val = Windows(val_path, args.block, cfg)
    switch = int(args.max_steps * args.long_from)
    per_step = args.batch_size * args.grad_accum * args.block
    per_long = args.long_batch_size * args.long_grad_accum * args.long_block
    print(f"schedule: {args.max_steps} steps, {per_step:,} tokens/step, {args.long_block} context "
          f"from step {switch}, total {(per_step * switch + per_long * (args.max_steps - switch))/1e9:.2f}B tokens",
          flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    def log(rec):
        if not args.probe:
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    log({"step": step, "event": "start", "tokens": tokens, "n_gpu": n_gpu})
    started = time.time()
    t0, tok0 = time.time(), tokens
    timed_from = None
    ooms = 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    while step < args.max_steps:
        if step >= switch:
            data, size, accum, block = long, args.long_batch_size, args.long_grad_accum, args.long_block
        else:
            data, size, accum, block = short, args.batch_size, args.grad_accum, args.block
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        stats = torch.zeros(len(STATS), device=device)
        unscaled = False
        try:
            for _ in range(accum):
                x = data.batch(size, device)
                with ctx:
                    out = model(x).float().mean(0)
                scaler.scale(out[0] / accum).backward()
                stats += out.detach() / accum
            scaler.unscale_(opt)
            unscaled = True
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if unscaled:
                scaler.update()
            opt.zero_grad(set_to_none=True)
            out = x = None
            torch.cuda.empty_cache()
            ooms += 1
            log({"step": step, "event": "oom", "count": ooms})
            print(f"out of memory at step {step} ({ooms} this session), step skipped", flush=True)
            if ooms >= 5:
                save(ckpt_path, model, opt, scaler, step, tokens, cfg, args)
                log({"step": step, "event": "oom_stop", "tokens": tokens})
                print("too many out-of-memory errors, saved and ending the session cleanly", flush=True)
                return
            continue
        step += 1
        tokens += size * accum * block

        if args.probe and step == 2:
            if device == "cuda":
                torch.cuda.synchronize()
            timed_from = (time.time(), tokens)

        if step % args.log_every == 0 and not args.probe:
            dt = time.time() - t0
            rec = {"step": step, **{k: round(v, 5) for k, v in zip(STATS, stats.tolist())},
                   "lr": lr, "grad_norm": round(float(grad_norm), 4), "scale": scaler.get_scale() if fp16 else 1.0,
                   "tok_per_s": round((tokens - tok0) / dt), "block": block, "tokens": tokens}
            print(f"step {step:>6}  loss {rec['loss']:.4f}  mlm {rec['mlm']:.3f}  rtd {rec['rtd']:.4f}  "
                  f"acc {rec['rtd_acc']:.4f}  gen {rec['gen_acc']:.3f}  lr {lr:.2e}  "
                  f"{rec['tok_per_s']/1e3:.1f}k tok/s  ctx {block}", flush=True)
            log(rec)
            t0, tok0 = time.time(), tokens

        if step % args.eval_every == 0 and not args.probe:
            v = evaluate(model, val, args.batch_size, args.eval_iters, device, ctx)
            rec = {"step": step, **{f"val_{k}": round(x, 5) for k, x in zip(STATS, v)}}
            print(f"  -> val mlm {rec['val_mlm']:.3f}  rtd {rec['val_rtd']:.4f}  acc {rec['val_rtd_acc']:.4f}", flush=True)
            log(rec)
            t0, tok0 = time.time(), tokens

        if step % args.ckpt_every == 0 and not args.probe:
            save(ckpt_path, model, opt, scaler, step, tokens, cfg, args)

        if args.max_hours and time.time() - started > args.max_hours * 3600:
            save(ckpt_path, model, opt, scaler, step, tokens, cfg, args)
            log({"step": step, "event": "session_limit", "tokens": tokens})
            print(f"session limit of {args.max_hours} h reached, saved at step {step}", flush=True)
            return

    if args.probe:
        if device == "cuda":
            torch.cuda.synchronize()
        t, tk = timed_from
        res = {"block": args.block, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
               "n_gpu": n_gpu, "tok_per_s": round((tokens - tk) / (time.time() - t)),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else 0.0}
        print("PROBE " + json.dumps(res), flush=True)
        return

    save(ckpt_path, model, opt, scaler, step, tokens, cfg, args)
    export(args.out / "disc.pt", model, cfg, step, tokens)
    log({"step": step, "event": "done", "tokens": tokens})
    print(f"done at step {step}, {tokens/1e9:.3f}B tokens; discriminator saved to {args.out / 'disc.pt'}", flush=True)


if __name__ == "__main__":
    main()
