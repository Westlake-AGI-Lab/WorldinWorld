"""Warped-source evidence injected into the self-attention of every DiT layer.

The K/V of the warped frames ("pseudo history") are computed by one extra forward pass and
appended to the attention context of the current chunk. Attention over the real context and
over each evidence bucket is computed separately with flash attention and merged exactly via
the log-sum-exp trick; a bucket collects all evidence tokens with the same visibility weight
w, whose logits are shifted by log(w) (partially visible tokens attract less attention).

EWA (evidence-weighted amplification): on the merged output the component that the evidence
adds relative to the context-only answer is extrapolated by a factor w_warp, restricted to the
part that agrees with the context answer (cos > 0) and clamped to the context norm. This is a
CFG-like amplification at attention-response level and costs no extra network forward.

CGAR source queries (see corr_bias) are merged into the same LSE sum as one more bucket.
"""
import logging
import math

import torch
from flash_attn import flash_attn_func
from wan.modules.attention import attention as wan_attention

from .corr_bias import CORR_PAD, resolve_positions, side_attention

DEAD_LOGW = -25.0


class PseudoHistBias:
    def __init__(self, num_layers, dsg=None):
        """dsg: dict(w_warp, steps, skip_last, tau) or None to disable EWA."""
        self.num_layers = num_layers
        self.sink_tokens = 0
        self._staged = None
        self._tok_logw = None
        self.ph_dsg = dsg
        self._dsg_skip = False
        self.corr_staged = None
        self._corr_step = 0
        self._corr_kv_s = None
        self._corr_sel_s = None
        self._corr_src2_cache = None
        self.log = []

    # -------------------------------------------------------------- attach / staging
    def attach(self, pipe):
        for i, blk in enumerate(pipe.model.blocks):
            blk.self_attn.layer_id = i
            blk.self_attn.kv_bank = self

    @staticmethod
    def detach(pipe):
        for blk in pipe.model.blocks:
            blk.self_attn.kv_bank = None

    def stage_kv(self, staged, tok_logw=None):
        """staged: per layer (k, v) of the selected evidence tokens; tok_logw: log weight per token."""
        self._staged = staged
        self._tok_logw = tok_logw if staged is not None else None

    def stage_corr(self, plan):
        self.corr_staged = plan
        self._corr_src2_cache = None
        if plan is None:
            self._corr_kv_s = self._corr_sel_s = None

    def get_staged(self, layer_id):
        if self._staged is None:
            return None, None
        return self._staged[layer_id]

    # -------------------------------------------------------------- hook entry point
    @torch.no_grad()
    def attend(self, layer_id, q, k_ctx, v_ctx):
        bank_k, bank_v = self.get_staged(layer_id)
        if bank_k is not None:
            return self.bias_attention(layer_id, q, k_ctx, v_ctx, bank_k, bank_v)
        if self.corr_staged is not None:
            return self.corr_only_attention(layer_id, q, k_ctx, v_ctx)
        return wan_attention(q, k_ctx, v_ctx)

    # -------------------------------------------------------------- CGAR source queries
    def _corr_src(self, layer_id):
        plan = self.corr_staged
        sq = plan.get("srcq") if plan is not None else None
        kv, sel = self._corr_kv_s, self._corr_sel_s
        if sq is None or kv is None or sel is None:
            return ()
        key = (int(self._corr_step), id(sq))
        c = self._corr_src2_cache
        if c is not None and c[0] == key:
            pw = c[1]
        else:
            pos, logw = resolve_positions(sq["cell"], sq["lg"], sel, sq["cell"].device)
            pw = None if pos is None else (pos, logw)
            if pw is not None and self._corr_step == 0:
                logging.info(f"[cgar] step0: {int((logw > CORR_PAD / 2).sum())} query tokens "
                             f"linked to source tokens ({int(sel.numel())} source tokens)")
            self._corr_src2_cache = (key, pw)
        if pw is None:
            return ()
        k_l, v_l = kv[layer_id]
        dev = pw[0].device
        return ((k_l.to(dev), v_l.to(dev), pw[0], pw[1]),)

    @torch.no_grad()
    def corr_only_attention(self, layer_id, q, k_ctx, v_ctx):
        src2 = self._corr_src(layer_id)
        o, l, _ = flash_attn_func(q, k_ctx, v_ctx, return_attn_probs=True)
        if not src2:
            return o
        outs, lses = [o.float()], [l.float()]
        for k2, v2, p2, w2 in src2:
            o_m, l_m = side_attention(q, k2, v2, p2, w2)
            outs.append(o_m)
            lses.append(l_m)
        m = torch.stack(lses).max(0).values
        ws = [torch.exp(le - m).permute(0, 2, 1).unsqueeze(-1) for le in lses]
        den = sum(ws)
        out = sum(o1 * w for o1, w in zip(outs, ws)) / den
        return out.to(q.dtype)

    # -------------------------------------------------------------- evidence attention
    @torch.no_grad()
    def bias_attention(self, layer_id, q, k_ctx, v_ctx, k_mem, v_mem):
        lg = self._tok_logw
        outs, lses = [], []
        L = k_ctx.shape[1]
        o, l, _ = flash_attn_func(q, k_ctx, v_ctx, return_attn_probs=True)
        outs.append(o.float())
        lses.append(l.float() + math.log(1.0))
        n_ctx = 1
        base = math.log(1.0)
        # one bucket per distinct token weight (sorted ascending), logits shifted by log(w)
        chan_ids = []
        for val in torch.unique(lg).tolist():
            if val < DEAD_LOGW:
                continue
            idx = (lg == val).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            o, l, _ = flash_attn_func(q, k_mem[:, idx], v_mem[:, idx], return_attn_probs=True)
            outs.append(o.float())
            lses.append(l.float() + base + float(val))
            chan_ids.append("warp")
        if self.corr_staged is not None:
            for k2, v2, p2, w2 in self._corr_src(layer_id):
                o_m2, l_m2 = side_attention(q, k2, v2, p2, w2)
                outs.append(o_m2)
                lses.append(l_m2)
                chan_ids.append("corr")
        m = torch.stack(lses).max(0).values
        ws = [torch.exp(le - m).permute(0, 2, 1).unsqueeze(-1) for le in lses]
        den = sum(ws)
        out = sum(o1 * w for o1, w in zip(outs, ws)) / den
        if self.ph_dsg is not None:
            out = self._apply_dsg(out, outs, ws, n_ctx, chan_ids)
        return out.to(q.dtype)

    def _apply_dsg(self, out, outs, ws, n_ctx, chan_ids):
        if self._dsg_skip or len(outs) <= n_ctx:
            return out
        cfg = self.ph_dsg
        w_warp = float(cfg.get("w_warp", 0.5))
        wc = sum(ws[i] for i in range(n_ctx))
        num_ctx = sum(outs[i] * ws[i] for i in range(n_ctx))
        o_ctx = num_ctx / wc.clamp(min=1e-30)
        wdt = torch.float32
        b32 = o_ctx.to(wdt)
        ids = [i for i in range(n_ctx, len(outs)) if chan_ids[i - n_ctx] == "warp"]
        if not ids or w_warp == 0.0:
            return out
        sc = sum(ws[i] for i in ids)
        g32 = ((num_ctx + sum(outs[i] * ws[i] for i in ids)) / (wc + sc).clamp(min=1e-30)).to(wdt)
        nb = b32.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        ng = g32.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        cos = ((g32 * b32).sum(-1, keepdim=True) / (ng * nb)).clamp(-1., 1.)
        coef = cos * (ng / nb)
        d = g32 - coef * b32
        d = d * cos.clamp(min=0.0)
        corr = d * w_warp
        new = out.to(wdt) + corr
        tau = cfg.get("tau", 1.0)
        if tau is not None:
            n_ref = b32.norm(dim=-1, keepdim=True)
            n_new = new.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            new = new * (float(tau) * n_ref / n_new).clamp(max=1.0)
        return new.to(out.dtype)
