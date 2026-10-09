"""Dense decoder-only transformer, its MoE counterpart, and the dense -> MoE conversion.

Dense FFN:  down(silu(gate x) * up x), bias-free, hidden width H.
MoE FFN:    shared(x) + routed_scale * sum_{e in top2} w_e * expert_e(x)
            shared  = dense FFN restricted to a random half A of the H neurons
            experts = 8 complementary pairs of random halves of the other half B
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 8192
    n_layer: int = 8
    d_model: int = 384
    n_head: int = 6
    ctx: int = 256
    ffn_hidden: int = 1024
    # MoE (only used when moe=True)
    shared_width: int = 512
    n_routed: int = 16
    routed_width: int = 256
    top_k: int = 2
    routed_scale: float = 2.0
    bias_gamma: float = 1e-3
    router_init_std: float = 0.02
    init_std: float = 0.02

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def rope_tables(ctx, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    ang = torch.outer(torch.arange(ctx, dtype=torch.float64), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):  # x: B, nh, T, hd
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[-2]
    c, s = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.nh, self.hd = c.n_head, c.d_model // c.n_head
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """Shared expert + top-k routed experts with loss-free bias balancing.

    Routed experts are stored stacked: w_gate/w_up [E, h, d], w_down [E, d, h].
    Only the routed (token, expert) pairs are computed: tokens are sorted by expert
    and each expert runs one matmul chain on its own slice.
    """

    def __init__(self, c: Config):
        super().__init__()
        E, h, d = c.n_routed, c.routed_width, c.d_model
        self.E, self.k, self.scale, self.gamma = E, c.top_k, c.routed_scale, c.bias_gamma
        self.shared = SwiGLU(d, c.shared_width)
        self.w_gate = nn.Parameter(torch.empty(E, h, d))
        self.w_up = nn.Parameter(torch.empty(E, h, d))
        self.w_down = nn.Parameter(torch.empty(E, d, h))
        self.router = nn.Linear(d, E, bias=False)
        # selection-only bias (persistent, part of the checkpoint) and per-step load counter
        self.register_buffer("route_bias", torch.zeros(E))
        self.register_buffer("step_load", torch.zeros(E), persistent=False)
        self.collect = False  # when True, accumulate routing statistics in self.stats
        self.aux_on, self.aux = False, None  # Switch aux loss fallback, off unless enabled by the trainer
        self.reset_stats()
        self.debug_token_hits = None  # tests: count how many expert evaluations each token gets

    def reset_stats(self):
        self.stats = {"load": torch.zeros(self.E), "entropy_sum": 0.0, "top1w_sum": 0.0, "tokens": 0}

    def route(self, x):  # x: N, d
        with torch.autocast(device_type=x.device.type, enabled=False):
            dt = torch.promote_types(x.dtype, torch.float32)  # router in fp32 (fp64 in tests)
            logits = F.linear(x.to(dt), self.router.weight.to(dt))
        sel = (logits + self.route_bias).topk(self.k, dim=-1).indices  # bias only for choosing
        w = torch.softmax(logits.gather(-1, sel), dim=-1)  # weights from raw logits, sum to 1
        return logits, sel, w

    def dispatch(self, x, sel, w):
        N, d = x.shape
        flat = sel.reshape(-1)  # N*k expert ids, token-major
        order = flat.argsort(stable=True)
        tok = order // self.k  # token id of each sorted assignment
        counts = torch.bincount(flat, minlength=self.E).tolist()
        xs = x[tok]
        wg, wu, wd = self.w_gate.unbind(0), self.w_up.unbind(0), self.w_down.unbind(0)
        outs, start = [], 0
        for e, n in enumerate(counts):
            if n == 0:
                continue
            xe = xs[start:start + n]
            outs.append((F.silu(xe @ wg[e].t()) * (xe @ wu[e].t())) @ wd[e].t())
            start += n
        ys = torch.cat(outs) * w.reshape(-1)[order].unsqueeze(-1)  # promotes bf16 -> fp32
        if self.debug_token_hits is not None:
            self.debug_token_hits += torch.bincount(tok, minlength=N).cpu()
        return torch.zeros(N, d, dtype=ys.dtype, device=x.device).index_add_(0, tok, ys)

    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, shp[-1])
        logits, sel, w = self.route(x)
        out = self.shared(x) + self.scale * self.dispatch(x, sel, w)
        with torch.no_grad():
            cnt = torch.bincount(sel.reshape(-1), minlength=self.E).float()
            if self.training:
                self.step_load += cnt
        if self.training and self.aux_on:
            # Switch-style balance loss E * sum_i f_i * P_i (fallback only; f has no gradient, P does)
            P = torch.softmax(logits, -1).mean(0)
            self.aux = self.E * (cnt / cnt.sum() * P).sum()
        with torch.no_grad():
            if self.collect:
                p = torch.softmax(logits, -1)
                self.stats["load"] += cnt.cpu()
                self.stats["entropy_sum"] += float(-(p * p.clamp_min(1e-12).log()).sum(-1).sum())
                self.stats["top1w_sum"] += float(w.max(-1).values.sum())
                self.stats["tokens"] += x.shape[0]
        return out.reshape(shp)

    @torch.no_grad()
    def update_bias(self):
        """b_i += gamma * sign(mean_load - load_i), load counted over the whole batch. Returns that batch's load."""
        load = self.step_load.clone()
        self.route_bias += self.gamma * torch.sign(load.mean() - load)
        self.step_load.zero_()
        return load


class Block(nn.Module):
    def __init__(self, c: Config, moe=False):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.attn = Attention(c)
        self.ffn = MoEFFN(c) if moe else SwiGLU(c.d_model, c.ffn_hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.ffn(self.n2(x))


class GPT(nn.Module):
    def __init__(self, c: Config, moe=False):
        super().__init__()
        self.cfg, self.moe = c, moe
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList([Block(c, moe) for _ in range(c.n_layer)])
        self.nf = RMSNorm(c.d_model)
        cos, sin = rope_tables(c.ctx, c.d_model // c.n_head)
        self.register_buffer("cos", cos.float(), persistent=False)
        self.register_buffer("sin", sin.float(), persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        c = self.cfg
        std, out_std = c.init_std, c.init_std / math.sqrt(2 * c.n_layer)
        nn.init.normal_(self.tok.weight, std=std)
        for b in self.blocks:
            nn.init.normal_(b.attn.qkv.weight, std=std)
            nn.init.normal_(b.attn.proj.weight, std=out_std)
            init_ffn(b.ffn, c)

    def moe_layers(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, MoEFFN)]

    def forward(self, idx, targets=None, reduction="mean"):
        h = self.tok(idx)
        for b in self.blocks:
            h = b(h, self.cos, self.sin)
        logits = F.linear(self.nf(h), self.tok.weight)  # tied head
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1), reduction=reduction)


def init_ffn(f, c: Config):
    std, out_std = c.init_std, c.init_std / math.sqrt(2 * c.n_layer)
    if isinstance(f, SwiGLU):
        nn.init.normal_(f.gate.weight, std=std)
        nn.init.normal_(f.up.weight, std=std)
        nn.init.normal_(f.down.weight, std=out_std)
    else:
        init_ffn(f.shared, c)
        nn.init.normal_(f.w_gate, std=std)
        nn.init.normal_(f.w_up, std=std)
        nn.init.normal_(f.w_down, std=out_std)
        nn.init.normal_(f.router.weight, std=c.router_init_std)
        f.route_bias.zero_()


# ----------------------------------------------------------------------------- conversion

def partition_indices(H, shared_width, n_routed, routed_width, gen):
    """A = shared neurons, experts = n_routed/2 complementary pairs of random halves of B."""
    perm = torch.randperm(H, generator=gen)
    A, B = perm[:shared_width].sort().values, perm[shared_width:]
    assert len(B) == 2 * routed_width, "each routed expert must hold exactly half of B"
    experts = []
    for _ in range(n_routed // 2):
        q = B[torch.randperm(len(B), generator=gen)]
        experts += [q[:routed_width].sort().values, q[routed_width:].sort().values]
    return A, experts


@torch.no_grad()
def upcycle(dense: GPT, seed: int):
    """Partition upcycling. Returns (moe model, per-layer index maps). Non-FFN tensors copied bit-exactly."""
    c = dense.cfg
    assert c.shared_width + 2 * c.routed_width == c.ffn_hidden
    moe = GPT(c, moe=True).to(dense.tok.weight.device, dense.tok.weight.dtype)
    dsd, msd = dense.state_dict(), moe.state_dict()
    for k, v in dsd.items():
        if ".ffn." not in k:
            msd[k].copy_(v)
    maps = []
    for l, (bd, bm) in enumerate(zip(dense.blocks, moe.blocks)):
        g = torch.Generator().manual_seed(seed * 1000 + l)
        A, experts = partition_indices(c.ffn_hidden, c.shared_width, c.n_routed, c.routed_width, g)
        f, m = bd.ffn, bm.ffn
        m.shared.gate.weight.copy_(f.gate.weight[A])
        m.shared.up.weight.copy_(f.up.weight[A])
        m.shared.down.weight.copy_(f.down.weight[:, A])
        for e, idx in enumerate(experts):
            m.w_gate[e].copy_(f.gate.weight[idx])
            m.w_up[e].copy_(f.up.weight[idx])
            m.w_down[e].copy_(f.down.weight[:, idx])
        # router: fresh small init (std 0.02), seeded per layer; bias starts at zero
        m.router.weight.copy_(torch.randn(m.router.weight.shape, generator=g) * c.router_init_std)
        m.route_bias.zero_()
        maps.append({"A": A, "experts": experts})
    return moe, maps


def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def transfer_optimizer_state(dense, dense_opt, moe, moe_opt, maps):
    """Copy AdamW state for unchanged tensors; slice dense FFN moments by the same neuron indices
    for shared and routed experts; routers get no state (fresh). Returns list of names with fresh state."""
    dstate = {n: dense_opt.state[p] for n, p in dense.named_parameters() if p in dense_opt.state}
    fresh = []
    for n, p in moe.named_parameters():
        if ".ffn." not in n:
            s = dstate[n]
            moe_opt.state[p] = {k: v.clone() for k, v in s.items()}
            continue
        l = int(n.split(".")[1])
        pre = f"blocks.{l}.ffn."
        A, experts = maps[l]["A"], maps[l]["experts"]
        sg, su, sd = dstate[pre + "gate.weight"], dstate[pre + "up.weight"], dstate[pre + "down.weight"]
        new = None
        if n.endswith("shared.gate.weight"):
            new = {k: (v[A].clone() if v.dim() else v.clone()) for k, v in sg.items()}
        elif n.endswith("shared.up.weight"):
            new = {k: (v[A].clone() if v.dim() else v.clone()) for k, v in su.items()}
        elif n.endswith("shared.down.weight"):
            new = {k: (v[:, A].clone() if v.dim() else v.clone()) for k, v in sd.items()}
        elif n.endswith("w_gate") or n.endswith("w_up"):
            src = sg if n.endswith("w_gate") else su
            new = {k: (torch.stack([v[i] for i in experts]) if v.dim() else v.clone()) for k, v in src.items()}
        elif n.endswith("w_down"):
            new = {k: (torch.stack([v[:, i] for i in experts]) if v.dim() else v.clone()) for k, v in sd.items()}
        if new is None:
            fresh.append(n)
        else:
            moe_opt.state[p] = new
    return fresh


# ----------------------------------------------------------------------------- accounting

def count_params(model: GPT):
    c = model.cfg
    emb = model.tok.weight.numel()
    attn = sum(p.numel() for b in model.blocks for p in b.attn.parameters())
    norms = sum(p.numel() for b in model.blocks for p in list(b.n1.parameters()) + list(b.n2.parameters())) + model.nf.weight.numel()
    out = {"embedding (tied head)": emb, "attention": attn, "norms": norms}
    if not model.moe:
        ffn = sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
        out["ffn"] = ffn
        out["total"] = emb + attn + norms + ffn
        out["active_per_token"] = out["total"]
        return out
    shared = sum(p.numel() for f in model.moe_layers() for p in f.shared.parameters())
    routed = sum(f.w_gate.numel() + f.w_up.numel() + f.w_down.numel() for f in model.moe_layers())
    router = sum(f.router.weight.numel() for f in model.moe_layers())
    out.update({"shared experts": shared, "routed experts": routed, "routers": router})
    out["total"] = emb + attn + norms + shared + routed + router
    out["active_per_token"] = emb + attn + norms + shared + router + routed * c.top_k // c.n_routed
    out["route_bias buffers (not trained by gradient)"] = sum(f.route_bias.numel() for f in model.moe_layers())
    assert out["total"] == sum(p.numel() for p in model.parameters())
    return out
