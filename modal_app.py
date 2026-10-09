"""Modal app for the dense -> MoE upcycling experiment.

  modal run modal_app.py::prep_data                       # tokenizer + tokenized TinyStories -> volume
  modal run modal_app.py::bench                           # 2-minute throughput benchmark, A10G and L40S
  modal run --detach modal_app.py::train --args "..."     # one training job (stage 1 or one stage-2 arm)

Everything persistent lives on the volume `s14-moe-upcycle`:
  /vol/data/   tokenizer.json, train.bin, val_fixed.bin, meta.json
  /vol/runs/<run>/  log.jsonl, latest.pt, <tag>_final.pt
"""
import os
import sys

import modal

app = modal.App("era-v5-s14-moe-upcycle")
vol = modal.Volume.from_name("s14-moe-upcycle", create_if_missing=True)
GPU = os.environ.get("S14_GPU", "A10G")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "numpy==2.4.4", "datasets==5.0.1", "tokenizers==0.23.1")
    .add_local_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"), remote_path="/root/src")
)


@app.function(image=image, cpu=8, memory=16384, volumes={"/vol": vol}, timeout=3600)
def prep_data():
    sys.path.insert(0, "/root/src")
    from data_prep import build_tinystories
    meta = build_tinystories("/vol/data")
    vol.commit()
    print(meta)
    return meta


@app.function(image=image, gpu="A10G", volumes={"/vol": vol}, timeout=900)
def bench_a10g():
    sys.path.insert(0, "/root/src")
    import bench
    return bench.run()


@app.function(image=image, gpu="L40S", volumes={"/vol": vol}, timeout=900)
def bench_l40s():
    sys.path.insert(0, "/root/src")
    import bench
    return bench.run()


@app.local_entrypoint()
def bench():
    import json
    calls = [bench_a10g.spawn(), bench_l40s.spawn()]
    res = [c.get() for c in calls]
    os.makedirs("results", exist_ok=True)
    with open("results/bench_throughput.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


@app.function(image=image, gpu=GPU, volumes={"/vol": vol}, timeout=4 * 3600,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def train(args: str):
    sys.path.insert(0, "/root/src")
    import torch
    from train import main
    print("gpu:", torch.cuda.get_device_name(0), "torch:", torch.__version__, flush=True)
    vol.reload()
    return main(args, commit=vol.commit)


@app.function(image=image, cpu=4, memory=16384, volumes={"/vol": vol}, timeout=1800)
def overlap_check():
    """Exact-duplicate check: how many validation stories (and how many of those inside the fixed eval set)
    also appear verbatim in the train split."""
    import hashlib
    import numpy as np
    from datasets import load_dataset
    from tokenizers import Tokenizer
    ds = load_dataset("roneneldan/TinyStories")
    h = lambda t: hashlib.blake2b(t.strip().encode(), digest_size=12).digest()
    train = set()
    for b in ds["train"].iter(batch_size=100_000):
        train.update(h(t) for t in b["text"] if t and t.strip())
    val = [t for t in ds["validation"]["text"] if t and t.strip()]
    # stories that make up the fixed eval set: walk the val stream until 2048*256+1 tokens are covered
    tok = Tokenizer.from_file("/vol/data/tokenizer.json")
    need, n_tok, n_fixed = 2048 * 256 + 1, 0, 0
    for t in val:
        if n_tok >= need:
            break
        n_tok += len(tok.encode(t.strip()).ids) + 1
        n_fixed += 1
    dup = [h(t) in train for t in val]
    res = {"train_unique_stories": len(train), "val_stories": len(val),
           "val_stories_exactly_in_train": int(sum(dup)),
           "fixed_eval_stories": n_fixed, "fixed_eval_stories_exactly_in_train": int(sum(dup[:n_fixed]))}
    print(res)
    return res


@app.local_entrypoint()
def overlap():
    import json
    res = overlap_check.remote()
    with open("results/train_val_overlap.json", "w") as f:
        json.dump(res, f, indent=2)


@app.function(image=image, cpu=4, memory=16384, volumes={"/vol": vol}, timeout=1800)
def build_clean_sets():
    sys.path.insert(0, "/root/src")
    from clean_eval import build_sets
    out = build_sets("/vol/data")
    vol.commit()
    return out


@app.function(image=image, gpu="A10G", volumes={"/vol": vol}, timeout=1800)
def clean_eval_gpu():
    sys.path.insert(0, "/root/src")
    from clean_eval import run
    vol.reload()
    return run("/vol/data", "/vol/runs")


@app.local_entrypoint()
def clean_eval():
    import json
    sets = build_clean_sets.remote()
    res = clean_eval_gpu.remote()
    with open("results/eval_by_train_overlap.json", "w") as f:
        json.dump({"sets": sets, "val_loss": res}, f, indent=2)
    print(json.dumps({"sets": sets, "val_loss": res}, indent=2))
