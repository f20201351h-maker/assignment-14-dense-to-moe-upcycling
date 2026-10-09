"""Re-evaluate the key checkpoints on validation sets split by train overlap.

TinyStories' validation split shares about 30% of its stories verbatim with the train split. The fixed eval set
used during training keeps that overlap (same for every arm). This script builds two extra 2048x256-token sets from
the validation split: `clean` (stories that never appear in train) and `dup` (stories that do), and evaluates
the dense checkpoint before conversion, the upcycled MoE at step 0, and the end of every stage-2 arm on
fixed / clean / dup.
"""
import hashlib
import json
import math
import os

import numpy as np
import torch


def build_sets(data_dir, n_windows=2048, ctx=256):
    done = os.path.join(data_dir, "val_split_sets.json")
    if os.path.exists(done):
        return json.load(open(done))
    from datasets import load_dataset
    from tokenizers import Tokenizer
    ds = load_dataset("roneneldan/TinyStories")
    h = lambda t: hashlib.blake2b(t.strip().encode(), digest_size=12).digest()
    train = set()
    for b in ds["train"].iter(batch_size=100_000):
        train.update(h(t) for t in b["text"] if t and t.strip())
    tok = Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
    val = [t.strip() for t in ds["validation"]["text"] if t and t.strip()]
    need = n_windows * ctx + 1
    out = {}
    for name, keep in (("clean", lambda t: h(t) not in train), ("dup", lambda t: h(t) in train)):
        ids, n = [], 0
        for t in val:
            if keep(t):
                e = tok.encode(t).ids + [0]
                ids.extend(e)
                n += 1
                if len(ids) >= need:
                    break
        assert len(ids) >= need, (name, len(ids))
        arr = np.array(ids[:need], dtype=np.uint16)
        arr.tofile(os.path.join(data_dir, f"val_{name}.bin"))
        out[name] = {"stories": n, "tokens_predicted": n_windows * ctx}
    with open(os.path.join(data_dir, "val_split_sets.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


@torch.no_grad()
def eval_tokens(model, arr, ctx=256, bs=128, device="cuda"):
    model.eval()
    n = (len(arr) - 1) // ctx
    ar = np.arange(ctx + 1)
    tot, cnt = 0.0, 0
    for i in range(0, n, bs):
        w = np.arange(i, min(i + bs, n))
        t = torch.from_numpy(arr[w[:, None] * ctx + ar].astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += float(model(t[:, :-1], t[:, 1:], reduction="sum"))
        cnt += t[:, 1:].numel()
    return tot / cnt


def run(data_dir, runs_dir, convert_seed=2026):
    from model import GPT, Config, upcycle
    sets = {k: np.fromfile(os.path.join(data_dir, f), dtype=np.uint16)
            for k, f in (("fixed", "val_fixed.bin"), ("clean", "val_clean.bin"), ("dup", "val_dup.bin"))}

    def load(path):
        ck = torch.load(path, map_location="cuda", weights_only=False)
        m = GPT(Config(**ck["cfg"]), moe=ck["moe"]).cuda()
        m.load_state_dict(ck["model"])
        return m, ck["step"]

    res = {}
    dense, step = load(os.path.join(runs_dir, "stage1", "stage1_final.pt"))
    res["dense_before_conversion"] = {"step": step, **{k: eval_tokens(dense, v) for k, v in sets.items()}}
    moe0, _ = upcycle(dense, convert_seed)
    res["moe_step0"] = {"step": step, **{k: eval_tokens(moe0, v) for k, v in sets.items()}}
    del dense, moe0
    for arm in ("stage2_dense", "stage2_moe", "stage2_moe_aux"):
        p = os.path.join(runs_dir, arm, f"{'stage2_dense' if arm == 'stage2_dense' else 'stage2_moe'}_final.pt")
        if os.path.exists(p):
            m, step = load(p)
            res[arm + "_end"] = {"step": step, **{k: eval_tokens(m, v) for k, v in sets.items()}}
            del m
    res["uniform_baseline"] = math.log(8192)
    return res
