"""Throughput benchmark: dense vs upcycled MoE forward+backward+AdamW step at the real config and batch."""
import time

import torch

from model import GPT, Config, count_params, param_groups, upcycle


def bench_one(model, B, T, V, steps, warm, compile_=False):
    dev = "cuda"
    if compile_:
        for b in model.blocks:
            b.attn.forward = torch.compile(b.attn.forward)
            f = b.ffn
            if hasattr(f, "shared"):
                f.shared.forward = torch.compile(f.shared.forward)
            else:
                f.forward = torch.compile(f.forward)
    opt = torch.optim.AdamW(param_groups(model, 0.1), lr=1e-4, betas=(0.9, 0.95), fused=True)
    g = torch.Generator(device=dev).manual_seed(0)
    layers = model.moe_layers()
    torch.cuda.reset_peak_memory_stats()
    for i in range(warm + steps):
        if i == warm:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        idx = torch.randint(0, V, (B, T + 1), device=dev, generator=g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(idx[:, :-1], idx[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for f in layers:
            f.update_bias()
        float(loss.detach())
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"tok_per_s": round(B * T * steps / dt), "sec_per_step": round(dt / steps, 4),
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def run(steps=25, warm=5, B=128):
    torch.backends.cuda.matmul.allow_tf32 = True
    c = Config()
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "batch_tokens": B * c.ctx}
    for comp in (False, True):
        torch.manual_seed(0)
        dense = GPT(c).cuda()
        moe, _ = upcycle(dense, 2026)
        out[f"dense{'_compiled' if comp else ''}"] = bench_one(dense, B, c.ctx, c.vocab_size, steps, warm, comp)
        del dense
        torch.cuda.empty_cache()
        out[f"moe{'_compiled' if comp else ''}"] = bench_one(moe, B, c.ctx, c.vocab_size, steps, warm, comp)
        del moe
        torch.cuda.empty_cache()
        print(out, flush=True)
    return out
