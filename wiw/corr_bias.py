"""Correspondence-guided attention (CGAR), source-query variant.

For every generated latent the planner projects the tracked 3D points of the matching source
frames (a) into the target camera and (b) into the source camera. Cells (16x16 px tokens) that
receive enough consistent votes get a partner cell in the clean source-frame block: the query
of the generated token is then given an additive logit bonus log(gamma) towards the key of its
partner token (`side_attention`), so the model looks at the same physical point at the same
moment in the source video.
"""
import math

import numpy as np
import torch

CORR_PAD = -1e30   # logit of an unused slot (exp underflows to exactly 0)


@torch.no_grad()
def side_attention(q, k_ph, v_ph, pos, logw):
    """Attention of q [B,Lq,H,D] against Ks selected slots of k_ph/v_ph [B,N,H,D].
    pos [Lq,Ks] slot indices, logw [Lq,Ks] additive logits (CORR_PAD = unused).
    Returns (out [B,Lq,H,D] float32, lse [B,H,Lq]) for LSE merging with flash attention."""
    B, Lq, H, D = q.shape
    assert pos.shape[0] == Lq, (pos.shape, Lq)
    N = k_ph.shape[1]
    valid = (logw > CORR_PAD / 2) & (pos >= 0) & (pos < N)
    pos_s = torch.where(valid, pos, torch.zeros_like(pos))
    scale = D ** -0.5
    qf = q.float()
    neg = torch.full_like(logw, CORR_PAD)
    logits = []
    for kk in range(pos.shape[1]):
        k_g = k_ph.index_select(1, pos_s[:, kk])
        l = (qf * k_g.float()).sum(-1) * scale
        l = l + torch.where(valid[:, kk], logw[:, kk], neg[:, kk])[None, :, None]
        logits.append(l)
    lg = torch.stack(logits, dim=-1)
    lse = torch.logsumexp(lg, dim=-1)
    p = torch.exp(lg - lse.unsqueeze(-1))
    out = None
    for kk in range(pos.shape[1]):
        v_g = v_ph.index_select(1, pos_s[:, kk]).float()
        t = p[..., kk].unsqueeze(-1) * v_g
        out = t if out is None else out + t
    return out, lse.transpose(1, 2).contiguous()


def resolve_positions(cell, lg_val, sel, device):
    """Map planned partner cells to slots of the staged source block (sel = sorted token ids).
    Returns (pos [Lq,2], logw [Lq,2]); the second slot is always unused."""
    Lq = cell.shape[0]
    pos = torch.zeros(Lq, 2, dtype=torch.long, device=device)
    logw = torch.full((Lq, 2), CORR_PAD, device=device)
    n = int(sel.numel())
    idx = torch.searchsorted(sel, cell.clamp(min=0))
    idx_c = idx.clamp(max=n - 1)
    hit = (cell >= 0) & (sel[idx_c] == cell)
    pos[:, 0] = idx_c
    logw[:, 0] = torch.where(hit, torch.full_like(logw[:, 0], lg_val), logw[:, 0])
    if int(hit.sum()) == 0:
        return None, None
    return pos, logw


class SourceQueryPlanner:
    """Builds, per chunk, the generated-token -> source-token correspondence map from
    <case>/tracks.npz (see wiw.track)."""

    def __init__(self, npz_path, device="cuda", gamma=2.0, vote_min=4, conf_min=0.15,
                 spread_max_tok=1.0, zband=0.05):
        d = np.load(npz_path)
        self._banks = {}
        for i, blo in enumerate(int(x) for x in d["bank_lo"]):
            self._banks[blo] = dict(v=torch.from_numpy(d[f"v_{i:02d}"].astype(np.float32)).to(device),
                                    conf=torch.from_numpy(d[f"conf_{i:02d}"].astype(np.float32)),
                                    f0=int(d["f0"][i]))
        self.T = int(d["T_global"])
        self.device = device
        self.gamma = float(gamma)
        self.vote_min = int(vote_min)
        self.conf_min = float(conf_min)
        self.spread_max_tok = float(spread_max_tok)
        self.zband = float(zband)
        self.v = self.conf = None
        self.P = self.V = 0
        self._f0 = 0
        self.log = []

    def _select_bank(self, lo):
        b = self._banks.get(int(lo))
        assert b is not None, f"tracks.npz has no bank for chunk lo={lo} (have {sorted(self._banks)})"
        self.v, self.conf, self._f0 = b["v"], b["conf"], b["f0"]
        self.P, self.V = int(self.v.shape[1]), int(self.v.shape[2])

    def _vf(self, f):
        assert 0 <= f - self._f0 < self.v.shape[0], (f, self._f0, self.v.shape[0])
        return self.v[f - self._f0]

    def _cf(self, f):
        return self.conf[f - self._f0]

    @staticmethod
    def lat_frames(g):
        return [0] if g == 0 else [4 * g - 3 + i for i in range(4)]

    def _fi(self, dw, f):
        return min(f, int(dw.c2ws.shape[0]) - 1, self.T - 1)

    @torch.no_grad()
    def _proj_frame(self, dw, f, gh, gw, w2c=None):
        if w2c is None:
            w2c = dw.w2cs[f]
        pts = self._vf(f).reshape(-1, 3)
        pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        u = dw.K[0, 0] * pc[:, 0] / z.clamp_min(1e-6) + dw.K[0, 2]
        vv = dw.K[1, 1] * pc[:, 1] / z.clamp_min(1e-6) + dw.K[1, 2]
        ok = (z > 1e-3) & (u >= 0) & (u < gw * 16) & (vv >= 0) & (vv < gh * 16)
        u_f = torch.nan_to_num(u, nan=0.0)
        vv_f = torch.nan_to_num(vv, nan=0.0)
        cell = ((vv_f.clamp(0, gh * 16 - 1) // 16).long() * gw + (u_f.clamp(0, gw * 16 - 1) // 16).long())
        zbuf = torch.full((gh * gw,), 1e9, device=z.device)
        zbuf.scatter_reduce_(0, cell, torch.where(ok, z, torch.full_like(z, 1e9)), reduce="amin")
        ok = ok & (z <= zbuf[cell] * (1.0 + self.zband))
        uv = torch.where(ok[:, None], torch.stack([u, vv], -1),
                         torch.full((pts.shape[0], 2), float("nan"), device=z.device))
        return uv.view(self.P, self.V, 2), ok.view(self.P, self.V)

    @torch.no_grad()
    def _proj_lat(self, dw, g, gh, gw, w2c=None):
        """Project the points of latent g (4 frames); each point keeps its last visible frame."""
        frs = [self._fi(dw, f) for f in self.lat_frames(g)]
        uvs, oks = [], []
        conf = torch.full((self.P,), 1e9)
        for f in frs:
            uv, ok = self._proj_frame(dw, f, gh, gw, w2c=w2c)
            uvs.append(uv)
            oks.append(ok)
            conf = torch.minimum(conf, self._cf(f).reshape(-1))
        okst = torch.stack(oks)
        uvst = torch.stack(uvs)
        fidx = torch.where(okst, torch.arange(len(frs), device=okst.device).view(-1, 1, 1).expand_as(okst),
                           torch.full_like(okst, -1, dtype=torch.long)).amax(0)
        vis = fidx >= 0
        uv = uvst.gather(0, fidx.clamp(min=0)[None, ..., None].expand(1, *fidx.shape, 2))[0]
        uv = torch.where(vis[..., None], uv, torch.full_like(uv, float("nan")))
        vis = vis & ~uv.isnan().any(-1)
        return uv, vis, conf, frs[-1]

    @torch.no_grad()
    def _pair_cells(self, gh, gw, uv_c, vis_c, conf_c, uv_h, vis_h, conf_h):
        """Vote partner cells: for each target cell, the median source-view position of the
        points that land in it (points must be visible in both views)."""
        out = []
        for p in range(self.P):
            if min(float(conf_c[p]), float(conf_h[p])) < self.conf_min:
                continue
            both = vis_c[p] & vis_h[p]
            if int(both.sum()) < self.vote_min:
                continue
            uc = uv_c[p][both]
            uh = uv_h[p][both]
            cells = ((uc[:, 1] // 16).long() * gw + (uc[:, 0] // 16).long())
            order = cells.argsort()
            cells, uh_o = cells[order], uh[order]
            uniq, cnt = torch.unique_consecutive(cells, return_counts=True)
            off = 0
            for ci, n in zip(uniq.tolist(), cnt.tolist()):
                sl = slice(off, off + n)
                off += n
                if n < self.vote_min:
                    continue
                med_h = uh_o[sl].median(0).values
                mad = (uh_o[sl] - med_h).abs().median(0).values
                if float(mad.max()) > self.spread_max_tok * 16.0:
                    continue
                tx = int(med_h[0].clamp(0, gw * 16 - 1) // 16)
                ty = int(med_h[1].clamp(0, gh * 16 - 1) // 16)
                out.append((ci, n, ty * gw + tx))
        return out

    @torch.no_grad()
    def plan_chunk(self, dw, lo, n_lat, gh, gw):
        """Plan for chunk starting at latent lo. Returns None when no cell has a partner."""
        self._select_bank(lo)
        fs = gh * gw
        Lq = n_lat * fs
        cur = [self._proj_lat(dw, lo + m, gh, gw) for m in range(n_lat)]
        srcq_cell = torch.full((Lq,), -1, dtype=torch.long)
        eye4 = torch.eye(4, device=dw.w2cs.device, dtype=dw.w2cs.dtype)
        n_srcq = 0
        for m in range(n_lat):
            uv_c, vis_c, conf_c, _ = cur[m]
            uv_s2, vis_s2, conf_s2, _ = self._proj_lat(dw, lo + m, gh, gw, w2c=eye4)
            best = {}
            for ci, n, ch in self._pair_cells(gh, gw, uv_c, vis_c, conf_c, uv_s2, vis_s2, conf_s2):
                if ci not in best or n > best[ci][0]:
                    best[ci] = (n, ch)
            for ci, (n, ch) in best.items():
                srcq_cell[m * fs + ci] = m * fs + ch
                n_srcq += 1
        if n_srcq == 0:
            return None
        rec = {"lo": lo, "corr_srcq": n_srcq}
        self.log.append(rec)
        return {"fs": fs, "n_lat": n_lat, "meta": rec,
                "srcq": {"cell": srcq_cell.to(self.device), "lg": math.log(max(self.gamma, 1e-9))}}
