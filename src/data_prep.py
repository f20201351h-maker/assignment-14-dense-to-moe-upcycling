"""Byte-level BPE (8192) on the TinyStories train split, then tokenize train and validation.

Outputs in out_dir:
  tokenizer.json   train.bin (uint16, stories joined by <|endoftext|>=0)
  val.bin          full validation stream
  val_fixed.bin    first n_val_windows*ctx+1 validation tokens: the fixed eval set used by every arm
  meta.json        counts and sha256 hashes
"""
import hashlib
import json
import os
import time

import numpy as np

EOT_STR = "<|endoftext|>"


def train_tokenizer(text_iter, length, vocab_size):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT_STR], min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(text_iter, trainer, length=length)
    assert tok.token_to_id(EOT_STR) == 0
    return tok


def encode_to_file(tok, batches, path):
    n_docs, n_tok = 0, 0
    with open(path, "wb") as f:
        for texts in batches:
            texts = [t.strip() for t in texts if t and t.strip()]
            encs = tok.encode_batch(texts)
            arr = np.fromiter((i for e in encs for i in (*e.ids, 0)), dtype=np.uint16)
            f.write(arr.tobytes())
            n_docs += len(texts)
            n_tok += len(arr)
    return n_docs, n_tok


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def build(train_batches_fn, n_train_docs, val_batches_fn, out_dir, vocab_size=8192, ctx=256, n_val_windows=2048):
    """train_batches_fn / val_batches_fn: zero-arg callables returning iterables of lists of strings."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    tok = train_tokenizer((t for b in train_batches_fn() for t in b), n_train_docs, vocab_size)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    t_tok = time.time() - t0
    tr_docs, tr_tok = encode_to_file(tok, train_batches_fn(), os.path.join(out_dir, "train.bin"))
    va_docs, va_tok = encode_to_file(tok, val_batches_fn(), os.path.join(out_dir, "val.bin"))
    val = np.fromfile(os.path.join(out_dir, "val.bin"), dtype=np.uint16)
    need = n_val_windows * ctx + 1
    assert len(val) >= need, (len(val), need)
    val[:need].tofile(os.path.join(out_dir, "val_fixed.bin"))
    meta = {
        "vocab_size": tok.get_vocab_size(), "eot_id": 0, "ctx": ctx,
        "train_docs": tr_docs, "train_tokens": tr_tok, "val_docs": va_docs, "val_tokens": va_tok,
        "val_fixed_windows": n_val_windows, "val_fixed_tokens_predicted": n_val_windows * ctx,
        "tokens_per_train_doc": tr_tok / max(1, tr_docs),
        "tokenizer_train_seconds": round(t_tok, 1), "total_seconds": round(time.time() - t0, 1),
        "sha256": {f: sha256(os.path.join(out_dir, f)) for f in ["tokenizer.json", "train.bin", "val_fixed.bin"]},
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def build_tinystories(out_dir, vocab_size=8192, ctx=256, n_val_windows=2048, batch=50_000):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories")
    tr, va = ds["train"], ds["validation"]

    def batches(split):
        return lambda: (b["text"] for b in split.iter(batch_size=batch))

    meta = build(batches(tr), len(tr), batches(va), out_dir, vocab_size, ctx, n_val_windows)
    meta["source"] = "roneneldan/TinyStories (HF default config: train / validation splits)"
    meta["hf_rows"] = {"train": len(tr), "validation": len(va)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta
