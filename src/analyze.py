"""Figures, headline numbers and the dense-vs-MoE table, computed only from the committed JSONL logs.

  python src/analyze.py --logs logs --out figures --results results

Expected logs: stage1_dense.jsonl, stage2_dense.jsonl, and one or more MoE arms
(stage2_moe.jsonl = bias balancing only; stage2_moe_aux.jsonl = bias + Switch aux loss fallback).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2, "lines.linewidth": 1.6,
})
C_S1, C_DENSE = "#555555", "#0072B2"  # Okabe-Ito
MOE_ARMS = [  # (log name, label, colour)
    ("stage2_moe", "upcycled MoE, bias balancing (as planned)", "#D55E00"),
    ("stage2_moe_aux", "upcycled MoE, bias + aux loss 0.01 (fallback)", "#009E73"),
]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def series(recs, typ, key):
    r = [x for x in recs if x["type"] == typ]
    return np.array([x["step"] for x in r]), np.array([x[key] for x in r], dtype=float)


def smooth(y, w):
    if len(y) < w:
        return y
    out = np.convolve(y, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), out])


def perf(recs, skip=2):
    p = [x for x in recs if x["type"] == "perf"][skip:]
    return float(np.median([x["tok_per_s"] for x in p])), max(x["peak_mem_gib"] or 0 for x in p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--results", default="results")
    ap.add_argument("--smooth", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.results, exist_ok=True)

    s1 = load(os.path.join(a.logs, "stage1_dense.jsonl"))
    dn = load(os.path.join(a.logs, "stage2_dense.jsonl"))
    arms = [(n, lab, c, load(os.path.join(a.logs, n + ".jsonl"))) for n, lab, c in MOE_ARMS
            if os.path.exists(os.path.join(a.logs, n + ".jsonl"))]
    cfg = next(x for x in s1 if x.get("event") == "config")
    tps = cfg["args"]["batch"] * cfg["cfg"]["ctx"]
    S1 = cfg["args"]["s1_steps"]
    total = S1 + cfg["args"]["s2_steps"]
    M = lambda steps: np.asarray(steps) * tps / 1e6  # tokens in millions

    es1, vs1 = series(s1, "eval", "val_loss")
    edn, vdn = series(dn, "eval", "val_loss")
    ts1, ls1 = series(s1, "train", "loss")
    tdn, ldn = series(dn, "train", "loss")
    dn_at = dict(zip(edn.tolist(), vdn.tolist()))
    pre = float(vs1[es1 == S1][0])

    # ------------------------------------------------------------------ headline numbers
    head = {"tokens_per_step": tps, "conversion_step": S1, "final_step": total,
            "val_before_conversion_dense": pre,
            "val_end_dense_control": float(vdn[edn == total][0]) if (edn == total).any() else None,
            "stage1_last_evals": [(int(s), round(float(v), 4)) for s, v in zip(es1, vs1)][-8:],
            "moe_arms": {}}
    for name, lab, _, mo in arms:
        emo, vmo = series(mo, "eval", "val_loss")
        moe0 = float(vmo[emo == S1][0])
        back = [(int(s), float(v)) for s, v in zip(emo, vmo) if s > S1 and v <= pre]
        h = {"label": lab,
             "val_step0_after_conversion": moe0, "conversion_jump": moe0 - pre,
             "first_eval_at_or_below_pre_conversion": None if not back else
                 {"step": back[0][0], "steps_after_conversion": back[0][0] - S1,
                  "tokens_after_conversion_M": (back[0][0] - S1) * tps / 1e6, "val_loss": back[0][1],
                  "dense_control_val_same_step": dn_at.get(back[0][0])},
             "val_end": float(vmo[emo == total][0]) if (emo == total).any() else None,
             "step0_controls": next(x for x in mo if x.get("event") == "step0_val")["val"],
             "minus_dense_at_matched_evals": {int(s): round(float(v) - dn_at[int(s)], 4) for s, v in zip(emo, vmo)
                                              if int(s) in dn_at and (int(s) % 1000 == 0 or int(s) in (S1, S1 + 100, S1 + 500, total))}}
        below = [int(s) for s, v in zip(emo, vmo) if int(s) in dn_at and v < dn_at[int(s)]]
        stays = [s for s in below if all(v < dn_at[int(t)] for t, v in zip(emo, vmo) if t >= s and int(t) in dn_at)]
        h["first_eval_below_dense_control"] = below[0] if below else None
        h["below_dense_control_from_step_onwards"] = stays[0] if stays else None
        if h["val_end"] is not None:
            h["end_below_pre_conversion"] = h["val_end"] < pre
            h["end_minus_pre_conversion"] = h["val_end"] - pre
            if head["val_end_dense_control"] is not None:
                h["end_minus_dense_control"] = h["val_end"] - head["val_end_dense_control"]
        rv = [x for x in mo if x["type"] == "routing" and x["source"] == "val"]
        keys = ("maxvio", "dead_zero", "dead_lt10pct", "min_share", "max_share", "entropy", "top1_weight")
        h["routing_val_step0"] = [{k: l[k] for k in keys} for l in rv[0]["layers"]]
        h["routing_val_end"] = [{k: l[k] for k in keys} for l in rv[-1]["layers"]]
        h["routing_val_end_step"] = rv[-1]["step"]
        # experts below 10% of uniform at every val eval in the last quarter of stage 2
        late = [x for x in rv if x["step"] >= S1 + 0.75 * (total - S1)]
        h["persistently_starved_experts_last_quarter"] = [
            int(np.sum(np.all(np.array([x["layers"][l]["load_share"] for x in late]) < 0.1 / len(rv[0]["layers"][l]["load_share"]), axis=0)))
            for l in range(len(rv[0]["layers"]))] if late else None
        head["moe_arms"][name] = h

    # ------------------------------------------------------------------ perf table
    conv = next(x for x in arms[0][3] if x.get("event") == "converted")
    pdn, pmo = conv["params_dense"], conv["params_moe"]
    tdn_s, mdn = perf(dn)
    ts1_s, ms1 = perf(s1)
    head["perf"] = {"stage1_dense": {"tok_s": ts1_s, "peak_gib": ms1}, "stage2_dense": {"tok_s": tdn_s, "peak_gib": mdn}}
    for name, _, _, mo in arms:
        t, m = perf(mo)
        head["perf"][name] = {"tok_s": t, "peak_gib": m}
    c = cfg["cfg"]
    hdr = "| | dense (control arm) | " + " | ".join(lab for _, lab, _, _ in arms) + " |\n"
    hdr += "|---|---:|" + "---:|" * len(arms) + "\n"
    rows = [
        ["total parameters", f"{pdn['total']:,}"] + [f"{pmo['total']:,} ({pmo['total'] / pdn['total']:.2f}x)"] * len(arms),
        ["active parameters per token", f"{pdn['active_per_token']:,}"] +
        [f"{pmo['active_per_token']:,} ({pmo['active_per_token'] / pdn['active_per_token']:.3f}x)"] * len(arms),
        ["FFN width per layer, total / active", f"{c['ffn_hidden']} / {c['ffn_hidden']}"] +
        [f"{c['shared_width'] + c['n_routed'] * c['routed_width']} / {c['shared_width'] + c['top_k'] * c['routed_width']}"] * len(arms),
        ["training tokens/s, median over stage 2 (A10G)", f"{tdn_s:,.0f}"] +
        [f"{head['perf'][n]['tok_s']:,.0f} ({head['perf'][n]['tok_s'] / tdn_s:.2f}x)" for n, _, _, _ in arms],
        ["peak GPU memory allocated, GiB", f"{mdn:.2f}"] +
        [f"{head['perf'][n]['peak_gib']:.2f} ({head['perf'][n]['peak_gib'] / mdn:.2f}x)" if mdn else "" for n, _, _, _ in arms],
    ]
    with open(os.path.join(a.results, "dense_vs_moe_table.md"), "w", encoding="utf-8") as f:
        f.write(hdr + "".join("| " + " | ".join(r) + " |\n" for r in rows))
    with open(os.path.join(a.results, "headline.json"), "w") as f:
        json.dump(head, f, indent=2)
    print(json.dumps(head, indent=2))

    # ------------------------------------------------------------------ fig 1: whole run
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    w = a.smooth
    ax.plot(M(ts1), smooth(ls1, w), color=C_S1, alpha=0.3, lw=1)
    ax.plot(M(tdn), smooth(ldn, w), color=C_DENSE, alpha=0.3, lw=1)
    ax.plot(M(es1), vs1, color=C_S1, marker="o", ms=2.5, label="stage 1: dense (val)")
    ax.plot(M(edn), vdn, color=C_DENSE, marker="o", ms=2.5, label="stage 2: dense continued, control (val)")
    lows = [vdn.min()]
    for name, lab, col, mo in arms:
        tm, lm = series(mo, "train", "loss")
        em, vm = series(mo, "eval", "val_loss")
        ax.plot(M(tm), smooth(lm, w), color=col, alpha=0.3, lw=1)
        ax.plot(M(em), vm, color=col, marker="o", ms=2.5, label=f"stage 2: {lab} (val)")
        lows.append(vm.min())
    ax.plot([], [], color="#999", alpha=0.5, lw=1, label=f"train loss, {w}-step moving mean (same colours)")
    ax.axvline(M(S1), color="#222", ls="--", lw=1)
    ax.text(M(S1) - 3, 1.95, "conversion ", ha="right", va="top", fontsize=9)
    ax.set_ylim(min(lows) - 0.05, 2.5)
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("cross-entropy loss (nats/token)")
    ax.set_title("TinyStories, 8 layers d384: dense, then dense vs upcycled MoE (y axis cut at 2.5)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.98))
    fig.savefig(os.path.join(a.out, "fig1_loss_full_run.png"))
    plt.close(fig)

    # inset-style second panel: stage 2 only, so the arms can be told apart
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    m2 = edn >= S1
    ax.plot(M(edn[m2]), vdn[m2], color=C_DENSE, marker="o", ms=2.5, label="dense continued, control")
    for name, lab, col, mo in arms:
        em, vm = series(mo, "eval", "val_loss")
        ax.plot(M(em), vm, color=col, marker="o", ms=2.5, label=lab)
    ax.axhline(pre, color="#222", lw=0.8, ls=":")
    ax.text(M(S1) + 2, pre + 0.003, f"val just before conversion {pre:.4f}", va="bottom", fontsize=8.5)
    ax.set_ylim(min(lows) - 0.02, pre + 0.12)
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("validation loss (nats/token)")
    ax.set_title("Stage 2 only: validation loss after conversion")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(a.out, "fig1b_stage2_val.png"))
    plt.close(fig)

    # ------------------------------------------------------------------ fig 2: conversion zoom
    lo, hi = S1 - 250, S1 + 1000
    sel = lambda s: (s >= lo) & (s <= hi)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    sw = 25
    ax.plot(ts1[sel(ts1)], smooth(ls1, sw)[sel(ts1)], color=C_S1, alpha=0.3, lw=1)
    ax.plot(tdn[sel(tdn)], smooth(ldn, sw)[sel(tdn)], color=C_DENSE, alpha=0.3, lw=1)
    win = [smooth(ls1, sw)[sel(ts1)], smooth(ldn, sw)[sel(tdn)]]
    ax.plot(es1[sel(es1)], vs1[sel(es1)], color=C_S1, marker="o", ms=3.5, label="dense before conversion (val)")
    ax.plot(edn[sel(edn)], vdn[sel(edn)], color=C_DENSE, marker="o", ms=3.5, label="dense continued (val)")
    win += [vs1[sel(es1)], vdn[sel(edn)]]
    for i, (name, lab, col, mo) in enumerate(arms):
        tm, lm = series(mo, "train", "loss")
        em, vm = series(mo, "eval", "val_loss")
        ax.plot(tm[sel(tm)], smooth(lm, sw)[sel(tm)], color=col, alpha=0.3, lw=1)
        win.append(smooth(lm, sw)[sel(tm)])
        ax.plot(em[sel(em)], vm[sel(em)], color=col, marker="o", ms=3.5, label=f"{lab} (val)")
        win.append(vm[sel(em)])
        h = head["moe_arms"][name]
        if i == 0:
            ax.annotate(f"MoE step 0 (val): {h['val_step0_after_conversion']:.4f}", (S1, h["val_step0_after_conversion"]),
                        xytext=(S1 - 250, h["val_step0_after_conversion"] - 0.015), fontsize=8.5,
                        arrowprops=dict(arrowstyle="-", color="#444", lw=0.8))
            fb = h["first_eval_at_or_below_pre_conversion"]
            if fb:
                ax.annotate(f"first eval at/below pre-conversion:\n+{fb['steps_after_conversion']} steps ({fb['val_loss']:.4f})",
                            (fb["step"], fb["val_loss"]), xytext=(fb["step"] - 120, fb["val_loss"] + 0.05), fontsize=8.5,
                            arrowprops=dict(arrowstyle="-", color="#444", lw=0.8))
    ax.plot([], [], color="#999", alpha=0.5, lw=1, label=f"train loss, {sw}-step moving mean (same colours)")
    ax.axhline(pre, color="#222", lw=0.8, ls=":")
    ax.text(lo + 5, pre - 0.003, f"pre-conversion val {pre:.4f}", va="top", fontsize=8.5)
    ax.axvline(S1, color="#222", ls="--", lw=1)
    win = np.concatenate(win)
    win = win[np.isfinite(win)]
    ax.set_ylim(win.min() - 0.01, win.max() + 0.01)
    ax.set_xlabel(f"global step (1 step = {tps:,} tokens; conversion at step {S1})")
    ax.set_ylabel("cross-entropy loss (nats/token)")
    ax.set_title("Zoom on the conversion")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(a.out, "fig2_conversion_zoom.png"))
    plt.close(fig)

    # ------------------------------------------------------------------ fig 3: routing health, one figure per arm
    for k, (name, lab, col, mo) in enumerate(arms):
        rv = [x for x in mo if x["type"] == "routing" and x["source"] == "val"]
        rt = [x for x in mo if x["type"] == "routing" and x["source"] == "train_interval"]
        L, E = len(rv[0]["layers"]), len(rv[0]["layers"][0]["load_share"])
        fig = plt.figure(figsize=(11.5, 7.8))
        gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 1], hspace=0.6, wspace=0.25)
        ax1, ax2 = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:])
        cmap = plt.get_cmap("viridis")
        st = np.array([x["step"] for x in rt]) - S1
        for l in range(L):
            ax1.plot(st, [x["layers"][l]["maxvio_batch_mean"] for x in rt], color=cmap(l / (L - 1)), lw=1.2, label=f"layer {l}")
            ax2.plot(st, [x["layers"][l]["dead_lt10pct"] for x in rt], color=cmap(l / (L - 1)), lw=1.2)
        for ax in (ax1, ax2):
            ax.set_xscale("symlog", linthresh=500)
            ax.set_xlabel("steps after conversion (symlog axis)")
        ax1.set_title("MaxVio per training batch (interval mean)", fontsize=10)
        ax1.legend(ncol=4, fontsize=7.5)
        ax2.set_title(f"experts below 10% of even share (of {E}), train batches", fontsize=10)
        ax2.set_ylim(-0.2, max(3, ax2.get_ylim()[1]))
        sv = np.array([x["step"] for x in rv]) - S1
        for l in range(L):
            axh = fig.add_subplot(gs[1 + l // 4, l % 4])
            share = np.array([x["layers"][l]["load_share"] for x in rv]).T * E  # 1.0 = even
            xe = np.concatenate([sv - 0.5 * np.diff(sv, prepend=sv[0] - 25), [sv[-1] + 0.5 * (sv[-1] - sv[-2])]])
            im = axh.pcolormesh(xe, np.arange(E + 1) - 0.5, share, cmap="RdBu_r", vmin=0, vmax=2, shading="flat")
            axh.invert_yaxis()
            axh.tick_params(axis="x", labelsize=7)
            axh.set_yticks([0, E - 1])
            axh.set_title(f"layer {l}", fontsize=8.5)
            axh.grid(False)
            if l % 4 == 0:
                axh.set_ylabel("expert")
            if l >= 4:
                axh.set_xlabel("steps after conversion", fontsize=7.5)
        cb = fig.colorbar(im, ax=fig.axes[2:], shrink=0.6, pad=0.01)
        cb.set_label("load share x 16 on the fixed val set (1 = even)")
        fig.suptitle(f"Routing health: {lab}", y=0.98)
        fig.savefig(os.path.join(a.out, f"fig3_routing_{name}.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
