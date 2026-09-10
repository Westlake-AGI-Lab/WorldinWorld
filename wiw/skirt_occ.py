"""Mesh-skirt occlusion for point-splat warping (after UniWorld-View's hybrid renderer).

At a depth discontinuity a triangulated depth map grows a stretched "skirt" triangle between
the foreground edge and the background behind it. Rasterised in the target view, the skirt
occludes everything behind it and its own footprint becomes a hole. Here the skirt is
emulated without a mesh renderer: every sliver pixel pair is projected into the target view
and its connecting segment is densely sampled into a hard z-buffer (`skirt_zbuf`); content
that is farther than the skirt is then discarded (`skirt_occlude`).
"""
import math

import torch
import torch.nn.functional as F

SLIVER_DEG = 2.5     # a pixel pair is a sliver when its displacement is within this angle of the view ray
TIE_EPS = 0.02       # depth tie tolerance: content wins ties
STEP_PX = 0.5        # sampling step along a projected segment (target pixels)
MAX_N = 1024         # samples per segment
CHUNK = 8192


@torch.no_grad()
def skirt_zbuf(rays, depth, w2c, K, out_hw=None, sliver_deg=SLIVER_DEG, step_px=STEP_PX,
               max_n=MAX_N, dilate=1):
    """Nearest skirt depth per target pixel (+inf where there is no skirt).

    rays (H,W,3): back-projected source pixel directions; depth (H,W): source z;
    w2c (4,4): source camera -> target camera; K (3,3): target intrinsics.
    Pixel convention as in depth_warp: uv = K Q / z - 0.5 (integer = pixel centre)."""
    H, W = depth.shape
    Ho, Wo = (H, W) if out_hw is None else tuple(out_hw)
    dev = depth.device
    P = rays * depth.unsqueeze(-1)
    Pa = torch.cat([P[:, :-1].reshape(-1, 3), P[:-1].reshape(-1, 3)], 0)
    Pb = torch.cat([P[:, 1:].reshape(-1, 3), P[1:].reshape(-1, 3)], 0)
    d = Pb - Pa
    mid = F.normalize(Pa + Pb, dim=-1, eps=1e-12)
    rad = (d * mid).sum(-1)
    lat = (d - rad.unsqueeze(-1) * mid).norm(dim=-1)
    ang = torch.atan2(lat, rad.abs().clamp_min(1e-12))
    sel = ang < math.radians(float(sliver_deg))
    idx_sel = sel.nonzero(as_tuple=True)[0]
    zbuf = torch.full((Ho * Wo,), float("inf"), device=dev)
    n_samples = 0
    if idx_sel.numel() > 0:
        R, t = w2c[:3, :3], w2c[:3, 3]
        Kt = K.T
        for lo in range(0, idx_sel.numel(), CHUNK):
            ii = idx_sel[lo:lo + CHUNK]
            Qa = Pa[ii] @ R.T + t
            Qb = Pb[ii] @ R.T + t
            za, zb = Qa[:, 2], Qb[:, 2]
            ok = (za > 1e-3) & (zb > 1e-3)
            if not bool(ok.any()):
                continue
            Qa, Qb, za, zb = Qa[ok], Qb[ok], za[ok], zb[ok]
            uva = (Qa @ Kt)[:, :2] / za.unsqueeze(-1) - 0.5
            uvb = (Qb @ Kt)[:, :2] / zb.unsqueeze(-1) - 0.5
            lo_xy = torch.minimum(uva, uvb)
            hi_xy = torch.maximum(uva, uvb)
            inside = ((hi_xy[:, 0] >= -1) & (lo_xy[:, 0] <= Wo)
                      & (hi_xy[:, 1] >= -1) & (lo_xy[:, 1] <= Ho))
            if not bool(inside.any()):
                continue
            uva, uvb, za, zb = uva[inside], uvb[inside], za[inside], zb[inside]
            L = (uvb - uva).norm(dim=-1)
            long_enough = L > 1.0
            if not bool(long_enough.any()):
                continue
            uva, uvb, za, zb, L = (uva[long_enough], uvb[long_enough],
                                   za[long_enough], zb[long_enough], L[long_enough])
            n = (L / float(step_px)).ceil().long().clamp(2, int(max_n))
            tot = int(n.sum())
            rep = torch.repeat_interleave(torch.arange(n.numel(), device=dev), n)
            starts = torch.cumsum(n, 0) - n
            k = torch.arange(tot, device=dev) - starts[rep]
            lam = (k.float() + 0.5) / n[rep].float()
            uv = uva[rep] + lam.unsqueeze(-1) * (uvb[rep] - uva[rep])
            z = 1.0 / ((1.0 - lam) / za[rep] + lam / zb[rep])
            px = torch.round(uv[:, 0]).long()
            py = torch.round(uv[:, 1]).long()
            # samples falling into either end-point pixel belong to the content, not the skirt
            pa = torch.round(uva[rep]).long()
            pb = torch.round(uvb[rep]).long()
            not_end = ~(((px == pa[:, 0]) & (py == pa[:, 1]))
                        | ((px == pb[:, 0]) & (py == pb[:, 1])))
            inb = (px >= 0) & (px < Wo) & (py >= 0) & (py < Ho) & not_end
            lin = (py[inb] * Wo + px[inb])
            zbuf.scatter_reduce_(0, lin, z[inb], reduce="amin")
            n_samples += int(inb.sum())
    zbuf = zbuf.view(Ho, Wo)
    if dilate > 0 and n_samples > 0:
        # close 1px gaps between adjacent segments (fill only pixels with skirt on both sides)
        inf = torch.tensor(float("inf"), device=dev)
        zp = F.pad(zbuf[None, None], (1, 1, 1, 1), value=float("inf"))[0, 0]
        l, r = zp[1:-1, :-2], zp[1:-1, 2:]
        u, dn = zp[:-2, 1:-1], zp[2:, 1:-1]
        gap = ~torch.isfinite(zbuf)
        cand_h = torch.where(torch.isfinite(l) & torch.isfinite(r), torch.maximum(l, r), inf)
        cand_v = torch.where(torch.isfinite(u) & torch.isfinite(dn), torch.maximum(u, dn), inf)
        zbuf = torch.where(gap, torch.minimum(cand_h, cand_v), zbuf)
    return zbuf


def skirt_occlude(zbuf, z_content, mask2, tie_eps=TIE_EPS):
    """Pixels killed by the skirt: a skirt exists and (the pixel is a hole already, or the
    skirt is closer than the splatted content by more than tie_eps)."""
    has = torch.isfinite(zbuf)
    return has & ((mask2 <= 0) | (zbuf < z_content * (1.0 - float(tie_eps))))
