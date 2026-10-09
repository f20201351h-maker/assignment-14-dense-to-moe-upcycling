"""Correctness checks for the dense -> MoE conversion (section 3 of the plan, checks 1-5 plus optimizer transfer).

Usage:
  python src/checks.py --out results/checks_random_init.txt                 # random dense model, real config
  python src/checks.py --ckpt stage1_final.pt --out results/checks_stage1.txt  # the real stage-1 checkpoint
"""
import argparse
import copy
import json
import sys

import torch
import torch.nn.functional as F

from model import GPT, Config, MoEFFN, count_params, param_groups, transfer_optimizer_state, upcycle


def ffn_hidden(f, x):
    return F.silu(x @ f.gate.weight.t()) * (x @ f.up.weight.t())


def run(dense, convert_seed, out, n_vec=4096, seed=0):
    P = lambda *s: (print(*s), out.append(" ".join(str(t) for t in s)))
    torch.set_grad_enabled(False)
    c = dense.cfg
    ok = True
    P(f"config: {json.dumps(c.to_dict())}")
    P(f"convert_seed: {convert_seed}")

    # ---------------- check 3: bit-identical carried tensors (in the model's own dtype, fp32)
    dense = dense.float().eval()
    moe32, maps = upcycle(dense, convert_seed)
    dsd, msd = dense.state_dict(), moe32.state_dict()
    carried = [k for k in dsd if ".ffn." not in k]
    same = [k for k in carried if torch.equal(dsd[k], msd[k])]
    new = [k for k in msd if k not in dsd]
    P(f"\n[3] carried tensors bit-identical: {len(same)}/{len(carried)} -> {'PASS' if len(same) == len(carried) else 'FAIL'}")
    P(f"    carried names (e.g.): {carried[:4]} ... {carried[-2:]}")
    P(f"    tensors that exist only in the MoE: {sorted(set(k.split('.', 2)[2] for k in new))} x {c.n_layer} layers")
    ok &= len(same) == len(carried)

    # ---------------- check 2: coverage of neuron sets
    P("\n[2] neuron coverage per layer")
    for l, m in enumerate(maps):
        A = set(m["A"].tolist())
        cnt = torch.zeros(c.ffn_hidden, dtype=torch.long)
        for idx in m["experts"]:
            cnt[idx] += 1
        B = set(range(c.ffn_hidden)) - A
        union = A | set(torch.cat(m["experts"]).tolist())
        sizes = sorted(set(len(i) for i in m["experts"]))
        distinct = len(set(tuple(i.tolist()) for i in m["experts"])) == len(m["experts"])
        pair_ok = all(sorted(m["experts"][2 * p].tolist() + m["experts"][2 * p + 1].tolist()) == sorted(B) for p in range(c.n_routed // 2))
        b_counts = cnt[sorted(B)]
        good = (len(union) == c.ffn_hidden and len(A) == c.shared_width and int(cnt[sorted(A)].sum()) == 0
                and int(b_counts.min()) == int(b_counts.max()) == c.n_routed // 2 and sizes == [c.routed_width] and distinct and pair_ok)
        ok &= good
        P(f"    layer {l}: |A|={len(A)} |union|={len(union)} A-neurons in routed experts={int(cnt[sorted(A)].sum())} "
          f"B-neuron multiplicity min/max={int(b_counts.min())}/{int(b_counts.max())} expert sizes={sizes} "
          f"all experts distinct={distinct} pairs complementary={pair_ok} -> {'PASS' if good else 'FAIL'}")

    # ---------------- check 1: experts equal restricted dense FFN (float64)
    dense64 = copy.deepcopy(dense).double()
    moe64, maps64 = upcycle(dense64, convert_seed)
    g = torch.Generator().manual_seed(seed)
    P("\n[1] expert == dense FFN restricted to its neuron set (float64, random inputs x ~ N(0,1), n=%d)" % n_vec)
    worst = 0.0
    for l, (bd, bm) in enumerate(zip(dense64.blocks, moe64.blocks)):
        x = torch.randn(n_vec, c.d_model, generator=g, dtype=torch.float64)
        f, m = bd.ffn, bm.ffn
        h = ffn_hidden(f, x)
        full = h @ f.down.weight.t()
        A, experts = maps64[l]["A"], maps64[l]["experts"]
        d_sh = (m.shared(x) - h[:, A] @ f.down.weight[:, A].t()).abs().max().item()
        d_ex = 0.0
        outs = []
        for e, idx in enumerate(experts):
            ye = (F.silu(x @ m.w_gate[e].t()) * (x @ m.w_up[e].t())) @ m.w_down[e].t()
            outs.append(ye)
            d_ex = max(d_ex, (ye - h[:, idx] @ f.down.weight[:, idx].t()).abs().max().item())
        d_pair = max((m.shared(x) + outs[2 * p] + outs[2 * p + 1] - full).abs().max().item() for p in range(c.n_routed // 2))
        scale = full.abs().max().item()
        worst = max(worst, d_sh, d_ex, d_pair)
        P(f"    layer {l}: max|shared - dense_A|={d_sh:.2e}  max|expert_e - dense_e| (16 experts)={d_ex:.2e}  "
          f"max|shared+pair_p - dense| (8 pairs)={d_pair:.2e}  (|dense out| max {scale:.2e})")
    ok &= worst < 1e-10
    P(f"    worst abs diff {worst:.2e} -> {'PASS' if worst < 1e-10 else 'FAIL'} (tolerance 1e-10 in float64)")

    # end-to-end: force every token onto complementary pair p with equal weights -> MoE logits == dense logits
    P("\n[1b] end-to-end: router zeroed, bias forces complementary pair p (weights 0.5/0.5, x2 scale) -> MoE logits vs dense logits (float64)")
    idx = torch.randint(0, c.vocab_size, (2, c.ctx), generator=g)
    ref = dense64(idx)
    forced = copy.deepcopy(moe64)
    for p in range(c.n_routed // 2):
        for f in forced.moe_layers():
            f.router.weight.zero_()
            f.route_bias.zero_()
            f.route_bias[2 * p] = f.route_bias[2 * p + 1] = 1.0
        d = (forced(idx) - ref).abs().max().item()
        worst = max(worst, d)
        P(f"    pair {p}: max|logits_moe - logits_dense| = {d:.2e}")
    ok &= worst < 1e-9
    P(f"    -> {'PASS' if worst < 1e-9 else 'FAIL'}")

    # ---------------- check 5: exactly top-2 executed per token, weights sum to 1, sparse == dense reference
    P("\n[5] routing: exactly k experts executed per token; weights sum to 1; sparse dispatch == masked dense reference (float64)")
    for l, bm in enumerate(moe64.blocks):
        m = bm.ffn
        m.router.weight.copy_(torch.randn(m.router.weight.shape, generator=g, dtype=torch.float64) * 0.5)  # spread routing
        x = torch.randn(n_vec, c.d_model, generator=g, dtype=torch.float64)
        logits, sel, w = m.route(x)
        m.debug_token_hits = torch.zeros(n_vec, dtype=torch.long)
        y_sparse = m.dispatch(x, sel, w)
        hits = m.debug_token_hits
        m.debug_token_hits = None
        dense_mask = torch.zeros(n_vec, m.E, dtype=torch.float64).scatter(1, sel, w)
        y_ref = torch.zeros_like(x)
        for e in range(m.E):
            ye = (F.silu(x @ m.w_gate[e].t()) * (x @ m.w_up[e].t())) @ m.w_down[e].t()
            y_ref += dense_mask[:, e:e + 1] * ye
        d = (y_sparse - y_ref).abs().max().item()
        distinct = bool((sel[:, 0] != sel[:, 1]).all())
        wsum = (w.sum(-1) - 1).abs().max().item()
        load = torch.bincount(sel.reshape(-1), minlength=m.E)
        good = int(hits.min()) == int(hits.max()) == c.top_k and distinct and wsum < 1e-12 and d < 1e-10 and int(load.sum()) == c.top_k * n_vec
        ok &= good
        if l < 2 or not good:
            P(f"    layer {l}: expert evaluations per token min/max={int(hits.min())}/{int(hits.max())}, total={int(load.sum())} (= {c.top_k} x {n_vec}), "
              f"selected experts distinct={distinct}, max|sum(w)-1|={wsum:.1e}, max|sparse-ref|={d:.1e} -> {'PASS' if good else 'FAIL'}")
    P(f"    (all {c.n_layer} layers checked)")

    # ---------------- optimizer state transfer
    P("\n[opt] AdamW state transfer (two dummy steps on the dense model, then convert)")
    d2 = copy.deepcopy(dense)
    opt = torch.optim.AdamW(param_groups(d2, 0.1), lr=1e-3)
    with torch.enable_grad():
        for _ in range(2):
            idx = torch.randint(0, c.vocab_size, (2, c.ctx + 1), generator=g)
            d2(idx[:, :-1], idx[:, 1:]).backward()
            opt.step()
            opt.zero_grad()
    m2, maps2 = upcycle(d2, convert_seed)
    mopt = torch.optim.AdamW(param_groups(m2, 0.1), lr=1e-3)
    fresh = transfer_optimizer_state(d2, opt, m2, mopt, maps2)
    dst = {n: opt.state[p] for n, p in d2.named_parameters()}
    mst = {n: mopt.state[p] for n, p in m2.named_parameters() if p in mopt.state}
    f0 = maps2[0]
    checks = [
        torch.equal(mst["tok.weight"]["exp_avg_sq"], dst["tok.weight"]["exp_avg_sq"]),
        torch.equal(mst["blocks.0.attn.qkv.weight"]["exp_avg"], dst["blocks.0.attn.qkv.weight"]["exp_avg"]),
        torch.equal(mst["blocks.0.ffn.shared.gate.weight"]["exp_avg"], dst["blocks.0.ffn.gate.weight"]["exp_avg"][f0["A"]]),
        torch.equal(mst["blocks.0.ffn.shared.down.weight"]["exp_avg_sq"], dst["blocks.0.ffn.down.weight"]["exp_avg_sq"][:, f0["A"]]),
        torch.equal(mst["blocks.0.ffn.w_up"]["exp_avg"][5], dst["blocks.0.ffn.up.weight"]["exp_avg"][f0["experts"][5]]),
        torch.equal(mst["blocks.0.ffn.w_down"]["exp_avg_sq"][9], dst["blocks.0.ffn.down.weight"]["exp_avg_sq"][:, f0["experts"][9]]),
    ]
    good = all(checks) and all(n.endswith("router.weight") for n in fresh) and len(fresh) == c.n_layer
    ok &= good
    P(f"    carried/sliced moments match: {checks} ; fresh state for {len(fresh)} tensors (routers only: "
      f"{all(n.endswith('router.weight') for n in fresh)}) -> {'PASS' if good else 'FAIL'}")

    # ---------------- check 4: parameter accounting
    P("\n[4] parameter accounting (counted from the real modules)")
    pd, pm = count_params(dense), count_params(moe32)
    for name, pdict in (("dense", pd), ("moe", pm)):
        P(f"    {name}: " + ", ".join(f"{k}={v:,}" for k, v in pdict.items()))
    P(f"    moe total / dense total = {pm['total'] / pd['total']:.3f}; moe active / dense = {pm['active_per_token'] / pd['total']:.4f}")
    P(f"    FFN width per layer: dense {c.ffn_hidden}; moe total {c.shared_width + c.n_routed * c.routed_width}, "
      f"active {c.shared_width + c.top_k * c.routed_width}")

    P(f"\nALL CHECKS: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--convert_seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    out = []
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        dense = GPT(Config(**ck["cfg"]))
        dense.load_state_dict(ck["model"])
        out.append(f"source: checkpoint {a.ckpt} (step {ck['step']})")
    else:
        dense = GPT(Config())
        out.append("source: randomly initialised dense model (seed 0), real configuration")
    print(out[0])
    ok = run(dense, a.convert_seed, out)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
