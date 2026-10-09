"""Stage 1: dense training. Stage 2: continue either the dense model (control) or its upcycled MoE.

Data order is stateless: global step s always reads windows perm[s*B:(s+1)*B] of the train token
stream, so both stage-2 arms see identical batches and a resumed run sees the same data.
"""
import argparse
import copy
import json
import math
import os
import shlex
import time

import numpy as np
import torch

from model import Config, GPT, MoEFFN, count_params, init_ffn, param_groups, transfer_optimizer_state, upcycle

EOT = 0


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True, choices=[1, 2])
    p.add_argument("--arm", default="dense", choices=["dense", "moe"])
    p.add_argument("--data_dir", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--stage1_ckpt", default="")
    p.add_argument("--s1_steps", type=int, default=6100)
    p.add_argument("--s2_steps", type=int, default=6100)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--peak_lr", type=float, default=1e-3)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337, help="data permutation and dense init")
    p.add_argument("--convert_seed", type=int, default=2026)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--dense_every", type=int, default=25)
    p.add_argument("--dense_before", type=int, default=200)
    p.add_argument("--dense_after", type=int, default=500)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--perf_every", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=0, help="stop this invocation after N updates (smoke tests)")
    p.add_argument("--n_val", type=int, default=0, help="use only the first n_val val windows (0 = all)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--compile", type=int, default=0)
    p.add_argument("--tiny", type=int, default=0, help="small model for CPU smoke tests")
    p.add_argument("--aux_coef", type=float, default=0.0,
                   help="Switch-style balance loss coefficient (approved fallback; 0 = bias balancing only)")
    return p.parse_args(argv)


def make_config(a):
    if a.tiny:
        return Config(vocab_size=8192, n_layer=2, d_model=64, n_head=2, ctx=64, ffn_hidden=128,
                      shared_width=64, n_routed=16, routed_width=32)
    return Config()


# ----------------------------------------------------------------------------- data

class Data:
    def __init__(self, data_dir, ctx, batch, seed, device, n_val=0):
        self.ctx, self.B, self.device = ctx, batch, device
        self.train = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
        val = np.fromfile(os.path.join(data_dir, "val_fixed.bin"), dtype=np.uint16)
        self.n_val = (len(val) - 1) // ctx if not n_val else n_val
        self.val = val
        self.n_windows = (len(self.train) - 1) // ctx
        self.perm = np.random.default_rng(seed).permutation(self.n_windows)
        self.ar = np.arange(ctx + 1)

    def max_steps(self):
        return self.n_windows // self.B

    def _to(self, t):
        t = torch.from_numpy(t.astype(np.int64))
        if self.device.startswith("cuda"):
            t = t.pin_memory().to(self.device, non_blocking=True)
        return t[:, :-1], t[:, 1:]

    def batch(self, step):
        w = self.perm[step * self.B:(step + 1) * self.B]
        assert len(w) == self.B, f"ran out of training data at step {step}"
        return self._to(self.train[w[:, None] * self.ctx + self.ar])

    def val_batches(self, bs):
        for i in range(0, self.n_val, bs):
            w = np.arange(i, min(i + bs, self.n_val))
            yield self._to(self.val[w[:, None] * self.ctx + self.ar])


# ----------------------------------------------------------------------------- helpers

def lr_at(step, a, total):
    if step < a.warmup:
        return a.peak_lr * (step + 1) / a.warmup
    frac = min(1.0, (step - a.warmup) / max(1, total - a.warmup))
    return a.peak_lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


def autocast(a):
    dev = "cuda" if a.device.startswith("cuda") else "cpu"
    return torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=bool(a.amp))


def routing_summary(load, E):
    """load: per-expert assignment counts. MaxVio = (max - mean)/mean (lesson section 10).
    Dead: zero load, and load below 10% of the uniform share (stated threshold)."""
    load = load.double()
    share = load / load.sum().clamp_min(1)
    mean = load.mean()
    return {
        "load_share": [round(float(x), 5) for x in share],
        "maxvio": float((load.max() - mean) / mean) if mean > 0 else None,
        "dead_zero": int((load == 0).sum()),
        "dead_lt10pct": int((share < 0.1 / E).sum()),
        "min_share": float(share.min()),
        "max_share": float(share.max()),
    }


@torch.no_grad()
def evaluate(model, data, a, bs=128):
    model.eval()
    layers = model.moe_layers()
    for f in layers:
        f.collect = True
        f.reset_stats()
    tot, n = 0.0, 0
    for x, y in data.val_batches(bs):
        with autocast(a):
            tot += float(model(x, y, reduction="sum"))
        n += y.numel()
    routing = None
    if layers:
        routing = []
        for f in layers:
            s = f.stats
            r = routing_summary(s["load"], f.E)
            r["entropy"] = s["entropy_sum"] / s["tokens"]
            r["top1_weight"] = s["top1w_sum"] / s["tokens"]
            r["bias"] = [round(float(b), 5) for b in f.route_bias.cpu()]
            routing.append(r)
            f.collect = False
    model.train()
    return tot / n, routing


@torch.no_grad()
def sample(model, tok, a, prompt="Once upon a time", n_new=160, temp=0.8, topk=40, seed=0):
    model.eval()
    dev = a.device
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.tensor([[EOT] + tok.encode(prompt).ids], device=dev)
    for _ in range(n_new):
        with autocast(a):
            logits = model(x[:, -model.cfg.ctx:])[:, -1, :].float() / temp
        v, i = logits.topk(topk)
        nxt = i.gather(-1, torch.multinomial(torch.softmax(v, -1), 1, generator=g))
        x = torch.cat([x, nxt], 1)
        if int(nxt) == EOT:
            break
    model.train()
    return tok.decode(x[0, 1:].tolist())


class Logger:
    def __init__(self, path, resume_step=None):
        self.path = path
        if resume_step is not None and os.path.exists(path):
            keep = [l for l in open(path, encoding="utf-8") if json.loads(l).get("step", -1) <= resume_step]
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(keep)
        elif os.path.exists(path):
            os.remove(path)
        self.f = open(path, "a", encoding="utf-8")

    def log(self, rec, echo=False):
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        if echo:
            print(json.dumps(rec)[:400], flush=True)


def save_ckpt(path, model, opt, step, a, extra=None):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "args": vars(a),
                "cfg": model.cfg.to_dict(), "moe": model.moe, **(extra or {})}, tmp)
    os.replace(tmp, path)


def make_opt(model, a):
    fused = a.device.startswith("cuda")
    return torch.optim.AdamW(param_groups(model, a.wd), lr=a.peak_lr, betas=(0.9, a.beta2), eps=1e-8, fused=fused)


def maybe_compile(model, a):
    if not a.compile:
        return
    for b in model.blocks:
        b.attn.forward = torch.compile(b.attn.forward)
        if isinstance(b.ffn, MoEFFN):
            b.ffn.shared.forward = torch.compile(b.ffn.shared.forward)
        else:
            b.ffn.forward = torch.compile(b.ffn.forward)


# ----------------------------------------------------------------------------- main

def main(argv=None, commit=lambda: None):
    if isinstance(argv, str):
        argv = shlex.split(argv)
    a = get_args(argv)
    os.makedirs(a.run_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = make_config(a)
    total = a.s1_steps + a.s2_steps
    start_step, end_step = (0, a.s1_steps) if a.stage == 1 else (a.s1_steps, total)
    data = Data(a.data_dir, cfg.ctx, a.batch, a.seed, a.device, a.n_val)
    assert total <= data.max_steps(), f"need {total} steps, data has {data.max_steps()}"
    tok_per_step = a.batch * cfg.ctx
    dense_lo, dense_hi = a.s1_steps - a.dense_before, a.s1_steps + a.dense_after
    eval_steps = set(range(0, total + 1, a.eval_every)) | {start_step, end_step, total}
    eval_steps |= set(range(max(0, dense_lo), dense_hi + 1, a.dense_every))

    latest = os.path.join(a.run_dir, "latest.pt")
    resume = os.path.exists(latest)
    ck = torch.load(latest, map_location=a.device, weights_only=False) if resume else None
    log = Logger(os.path.join(a.run_dir, "log.jsonl"), resume_step=ck["step"] if resume else None)
    tag = f"stage{a.stage}" + ("" if a.stage == 1 else f"_{a.arm}")
    is_moe = a.stage == 2 and a.arm == "moe"

    tokenizer = None
    tp = os.path.join(a.data_dir, "tokenizer.json")
    if os.path.exists(tp):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tp)

    if resume:
        model = GPT(cfg, moe=is_moe).to(a.device)
        model.load_state_dict(ck["model"])
        opt = make_opt(model, a)
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        log.log({"type": "event", "step": step, "event": "resumed", "from": latest}, echo=True)
    elif a.stage == 1:
        torch.manual_seed(a.seed)
        model = GPT(cfg).to(a.device)
        opt = make_opt(model, a)
        step = 0
        log.log({"type": "event", "step": 0, "event": "config", "tag": tag, "args": vars(a), "cfg": cfg.to_dict(),
                 "params": count_params(model), "train_tokens_available": int(len(data.train)),
                 "val_tokens_fixed": int(data.n_val * cfg.ctx)}, echo=True)
    else:
        s1 = torch.load(a.stage1_ckpt, map_location=a.device, weights_only=False)
        assert s1["step"] == a.s1_steps, (s1["step"], a.s1_steps)
        dense = GPT(cfg).to(a.device)
        dense.load_state_dict(s1["model"])
        dopt = make_opt(dense, a)
        dopt.load_state_dict(s1["opt"])
        step = a.s1_steps
        log.log({"type": "event", "step": step, "event": "config", "tag": tag, "args": vars(a), "cfg": cfg.to_dict(),
                 "stage1_ckpt": a.stage1_ckpt, "stage1_step": s1["step"]}, echo=True)
        if not is_moe:
            model, opt = dense, dopt
            log.log({"type": "event", "step": step, "event": "params", "params": count_params(model)}, echo=True)
        else:
            dense_val, _ = evaluate(dense, data, a)
            model, maps = upcycle(dense, a.convert_seed)
            opt = make_opt(model, a)
            fresh = transfer_optimizer_state(dense, dopt, model, opt, maps)
            log.log({"type": "event", "step": step, "event": "converted", "convert_seed": a.convert_seed,
                     "params_dense": count_params(dense), "params_moe": count_params(model),
                     "optimizer_state": "carried for unchanged tensors; sliced by neuron index for shared/routed experts; fresh for: " + ", ".join(fresh)}, echo=True)
            # step-0 controls on the fixed validation set (no training)
            step0 = {"dense_checkpoint": dense_val}
            step0["upcycled_moe"], r0 = evaluate(model, data, a)
            ctl = copy.deepcopy(model)
            torch.manual_seed(a.convert_seed + 1)
            for f in ctl.moe_layers():  # routed experts random, shared expert / router / rest inherited
                c = cfg
                torch.nn.init.normal_(f.w_gate, std=c.init_std)
                torch.nn.init.normal_(f.w_up, std=c.init_std)
                torch.nn.init.normal_(f.w_down, std=c.init_std / math.sqrt(2 * c.n_layer))
            step0["moe_random_routed_experts"], _ = evaluate(ctl, data, a)
            for f in ctl.moe_layers():  # shared expert random as well (all FFN weights random)
                init_ffn(f.shared, cfg)
            step0["moe_random_all_experts"], _ = evaluate(ctl, data, a)
            del ctl
            torch.manual_seed(a.convert_seed + 2)
            rnd = GPT(cfg, moe=True).to(a.device)
            step0["fully_random_moe"], _ = evaluate(rnd, data, a)
            del rnd
            step0["uniform_baseline_ln_vocab"] = math.log(cfg.vocab_size)
            log.log({"type": "event", "step": step, "event": "step0_val", "val": step0}, echo=True)
            del dense, dopt

    maybe_compile(model, a)
    model.train()
    layers = model.moe_layers()
    for f in layers:
        f.aux_on = a.aux_coef > 0
    E = cfg.n_routed
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def do_eval(s):
        vl, routing = evaluate(model, data, a)
        log.log({"type": "eval", "step": s, "tokens": s * tok_per_step, "val_loss": vl, "tag": tag}, echo=True)
        if routing is not None:
            log.log({"type": "routing", "source": "val", "step": s, "layers": routing})

    if not resume:
        do_eval(step)
        commit()

    # interval accumulators for training-batch routing health
    def new_acc():
        return {"load": torch.zeros(len(layers), E, device=a.device), "maxvio_sum": torch.zeros(len(layers), device=a.device), "n": 0}
    acc = new_acc()
    perf_t, perf_steps = 0.0, 0
    stop_at = end_step if not a.max_steps else min(end_step, step + a.max_steps)
    params = [p for p in model.parameters()]

    while step < stop_at:
        t0 = time.perf_counter()
        x, y = data.batch(step)
        lr = lr_at(step, a, total)
        for g in opt.param_groups:
            g["lr"] = lr
        with autocast(a):
            loss = model(x, y)
        aux = torch.stack([f.aux for f in layers]).mean() if layers and a.aux_coef > 0 else None
        (loss + a.aux_coef * aux if aux is not None else loss).backward()
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if layers:  # loss-free bias update from this batch's load; accumulate health stats on device
            ld = torch.stack([f.update_bias() for f in layers])
            acc["load"] += ld
            acc["maxvio_sum"] += (ld.max(1).values - ld.mean(1)) / ld.mean(1)
            acc["n"] += 1
        lval = float(loss.detach())
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        perf_t += time.perf_counter() - t0
        perf_steps += 1
        step += 1
        rec = {"type": "train", "step": step, "tokens": step * tok_per_step, "loss": lval, "lr": lr, "gnorm": float(gn)}
        if aux is not None:
            rec["aux"] = float(aux.detach())  # "loss" stays the pure LM cross-entropy
        log.log(rec)
        if not math.isfinite(lval):
            log.log({"type": "event", "step": step, "event": "non-finite loss, stopping"}, echo=True)
            commit()
            raise RuntimeError("non-finite loss")
        if perf_steps == a.perf_every or step == stop_at:
            mem = torch.cuda.max_memory_allocated() / 2**30 if a.device.startswith("cuda") else None
            log.log({"type": "perf", "step": step, "tok_per_s": perf_steps * tok_per_step / perf_t,
                     "sec_per_step": perf_t / perf_steps, "peak_mem_gib": mem, "window": perf_steps}, echo=(step % 500 == 0))
            perf_t, perf_steps = 0.0, 0
        if step in eval_steps:
            if layers and acc["n"]:
                tr = []
                for i in range(len(layers)):
                    r = routing_summary(acc["load"][i].cpu(), E)
                    r["maxvio_batch_mean"] = float(acc["maxvio_sum"][i] / acc["n"])
                    tr.append(r)
                log.log({"type": "routing", "source": "train_interval", "step": step, "n_steps": acc["n"], "layers": tr})
                acc = new_acc()
            do_eval(step)
            commit()
        if step % a.ckpt_every == 0 or step == end_step:
            save_ckpt(latest, model, opt, step, a)
            commit()

    if step == end_step:
        final = os.path.join(a.run_dir, f"{tag}_final.pt")
        save_ckpt(final, model, opt, step, a)
        if tokenizer is not None:
            for sd in (0, 1):
                log.log({"type": "sample", "step": step, "seed": sd, "text": sample(model, tokenizer, a, seed=sd)}, echo=True)
        log.log({"type": "event", "step": step, "event": "done", "final_ckpt": final}, echo=True)
        commit()
    return step


if __name__ == "__main__":
    main()
