"""Dense point tracks for correspondence-guided attention (CGAR).

  python -m wiw.track --case tennis

For every generated chunk a bank of tracks is built with CoWTracker on the window of source
frames that the chunk can attend to (its own 16 frames + the 20 preceding ones). Tracks are
seeded every 4 frames on a dense grid (plus extra points in high-motion cells), de-duplicated,
back-projected with the case depth into 3D (per-frame source-camera coordinates), and stored
in <case>/tracks.npz (keys bank_lo, f0, T_global, v_XX [Tw,1,N,3] fp16 with NaN = invisible,
conf_XX). Requires the CoWTracker repository (WIW_COW_REPO) and checkpoint (WIW_COW_CKPT).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

from .config import COW_REPO, COW_CKPT, case_dir
from .depth_warp import DepthWarper
from .traj import K4_frame

H2, W2 = 448, 784          # CoWTracker input (multiple of 112)
STRIDE_R = 4               # grid stride in resized pixels
MARGIN_R = 6
VIS_TH = 0.5
DEDUP_PX = 6.0
EDGE_HARD, EDGE_SOFT, Z_MIN, TAP_R = 1.5, 1.2, 0.05, 2
GATE_P50, GATE_P95 = 0.5, 2.0


def bank_table(T):
    """Per chunk: (first latent lo, window first/last frame f0/f1, seed frames sa..sb)."""
    t_lat = (T - 1) // 4 + 1
    n_ck = t_lat // 4
    if n_ck * 4 + 2 > t_lat:
        n_ck -= 1
    rows = []
    for c in range(n_ck):
        lo = 4 * c
        f0 = max(0, 4 * (lo - 4) - 3)
        f1 = 4 * lo + 12
        assert f1 <= T - 1
        rows.append((lo, f0, f1, max(0, 4 * lo - 3), f1))
    return rows


def dedupe_tracks(track2d, vis, queries, r=DEDUP_PX):
    """Drop re-seeded tracks that lie within r px of an already kept visible track."""
    N = track2d.shape[1]
    order = np.argsort(queries[:, 0], kind="stable")
    keep = np.ones(N, bool)
    kept = []
    for tq in np.unique(queries[:, 0]):
        grp = order[queries[order, 0] == tq]
        if not kept:
            kept.extend(grp.tolist())
            continue
        ka = np.asarray(kept)
        f = int(tq)
        ref = track2d[f, ka]
        ref_ok = vis[f, ka] > VIS_TH
        for j in grp:
            d = np.linalg.norm(ref[ref_ok] - track2d[f, j], axis=1)
            if d.size and d.min() < r:
                keep[j] = False
            else:
                kept.append(int(j))
    return keep


def _gather_z(depth_t, u, v):
    Hh, Ww = depth_t.shape
    uc = u.clamp(0, Ww - 1.001)
    vc = v.clamp(0, Hh - 1.001)
    x0, y0 = uc.floor().long(), vc.floor().long()
    x1, y1 = (x0 + 1).clamp(max=Ww - 1), (y0 + 1).clamp(max=Hh - 1)
    wx, wy = uc - x0.float(), vc - y0.float()
    d = depth_t
    z_bi = (d[y0, x0] * (1 - wx) * (1 - wy) + d[y0, x1] * wx * (1 - wy)
            + d[y1, x0] * (1 - wx) * wy + d[y1, x1] * wx * wy)
    taps = []
    for dy in (-TAP_R, 0, TAP_R):
        for dx in (-TAP_R, 0, TAP_R):
            xi = (uc + dx).round().long().clamp(0, Ww - 1)
            yi = (vc + dy).round().long().clamp(0, Hh - 1)
            taps.append(d[yi, xi])
    return z_bi, torch.stack(taps, -1)


def lift_tracks(track2d, vis, depth, keep, K):
    """Back-project 2D tracks with the source depth. Points on depth discontinuities or
    invalid depth are dropped (NaN). Returns v3 [T,N,3]."""
    T, N, _ = track2d.shape
    Hh, Ww = depth.shape[-2:]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v3 = torch.full((T, N, 3), float("nan"), dtype=torch.float32, device=depth.device)
    for f in range(T):
        u, vv = track2d[f, :, 0], track2d[f, :, 1]
        act = (vis[f] > VIS_TH) & (u >= 0) & (u <= Ww - 1) & (vv >= 0) & (vv <= Hh - 1)
        if keep is not None:
            act = act & keep[f][vv.round().long().clamp(0, Hh - 1), u.round().long().clamp(0, Ww - 1)]
        if not bool(act.any()):
            continue
        z_bi, taps = _gather_z(depth[f], u[act], vv[act])
        tmin, tmax = taps.min(-1).values, taps.max(-1).values
        bad_z = (z_bi <= Z_MIN) | (tmin <= Z_MIN)
        ratio = tmax / tmin.clamp(min=1e-6)
        hard = (ratio > EDGE_HARD) & ~bad_z
        soft = (ratio > EDGE_SOFT) & ~hard & ~bad_z
        z = torch.where(soft, taps.median(-1).values, z_bi)
        idx = act.nonzero(as_tuple=True)[0]
        good = ~hard & ~bad_z
        gi, zg = idx[good], z[good]
        v3[f, gi, 0] = (u[gi] - cx) / fx * zg
        v3[f, gi, 1] = (vv[gi] - cy) / fy * zg
        v3[f, gi, 2] = zg
    return v3


def roundtrip(v3, track2d, K):
    m = ~torch.isnan(v3[..., 2])
    if not bool(m.any()):
        return dict(p50=float("nan"), p95=float("nan"), n=0)
    x, y, z = v3[..., 0][m], v3[..., 1][m], v3[..., 2][m]
    e = torch.stack([K[0, 0] * x / z + K[0, 2] - track2d[..., 0][m],
                     K[1, 1] * y / z + K[1, 2] - track2d[..., 1][m]], -1).norm(dim=-1).float()
    return dict(p50=float(e.quantile(0.5)), p95=float(e.quantile(0.95)), n=int(m.sum()))


def _resize_win(frames_win):
    v = np.stack([cv2.resize(f, (W2, H2), interpolation=cv2.INTER_AREA) for f in frames_win])
    return torch.from_numpy(v).permute(0, 3, 1, 2).to("cuda", torch.float16)


def hot_extra(frames_win, sx, sy, gh, gw):
    """Extra seed points (4 per cell) in the 25% most dynamic 16px cells of the window."""
    f = frames_win.astype(np.int16)
    diff = np.abs(f[1:] - f[:-1]).mean(-1)
    mc = diff.reshape(diff.shape[0], gh, 16, gw, 16).mean((2, 4)).mean(0)
    hy, hx = np.nonzero(mc >= np.quantile(mc, 0.75))
    off = np.array([[4, 4], [4, 12], [12, 4], [12, 12]], np.float32)
    pts = (np.stack([hx, hy], -1)[:, None] * 16.0 + off[None]).reshape(-1, 2)
    ex = torch.as_tensor(np.clip(pts[:, 0] / sx, 0, None), device="cuda").round().long().clamp(0, W2 - 1)
    ey = torch.as_tensor(np.clip(pts[:, 1] / sy, 0, None), device="cuda").round().long().clamp(0, H2 - 1)
    return ey, ex


@torch.no_grad()
def cow_track_bank(model, win_t, t_list, gy, gx, sx, sy):
    """Dense tracks from each seed frame, forward and (reversed) backward."""
    Tw, n = win_t.shape[0], gy.numel()
    t2s, vis, qs = [], [], []
    for t in t_list:
        tr = torch.empty(Tw, n, 2, device="cuda")
        vc = torch.empty(Tw, n, device="cuda")
        if t > 0:
            o = model(win_t[:t + 1].flip(0))
            tr[:t + 1] = o["track"][0].float().flip(0)[:, gy, gx, :]
            vc[:t + 1] = (o["vis"][0].float() * o["conf"][0].float()).flip(0)[:, gy, gx]
        o = model(win_t[t:])
        tr[t:] = o["track"][0].float()[:, gy, gx, :]
        vc[t:] = (o["vis"][0].float() * o["conf"][0].float())[:, gy, gx]
        vc[t] = 1.0
        tr[..., 0] *= sx
        tr[..., 1] *= sy
        t2s.append(tr.cpu().numpy())
        vis.append(vc.cpu().numpy())
        qs.append(np.stack([np.full(n, t, np.float32), (gx.cpu().numpy() * sx).astype(np.float32),
                            (gy.cpu().numpy() * sy).astype(np.float32)], -1))
    return np.concatenate(t2s, 1), np.concatenate(vis, 1), np.concatenate(qs, 0)


def run(case, force=False, vis_th=VIS_TH):
    out = os.path.join(case, "tracks.npz")
    if os.path.exists(out) and not force:
        print(f"[track] {out} exists, skip")
        return
    frames = np.load(os.path.join(case, "frames.npy"))
    T, Hh, Ww, _ = frames.shape
    gh, gw = Hh // 16, Ww // 16
    sx, sy = Ww / W2, Hh / H2
    disp = np.load(os.path.join(case, "depth.npz"))["disp"].astype(np.float32)
    dw = DepthWarper(disp, K4_frame(), np.tile(np.eye(4), (T, 1, 1)), device="cuda", depth_smooth=True)
    dep, kee, K_t = dw.depth.detach().float(), getattr(dw, "keep", None), dw.K.detach().float()
    sys.path.insert(0, COW_REPO)
    from cowtracker import CoWTracker
    model = CoWTracker.from_checkpoint(COW_CKPT, device="cuda", dtype=torch.float16)
    ys = torch.arange(MARGIN_R, H2 - MARGIN_R, STRIDE_R, device="cuda")
    xs = torch.arange(MARGIN_R, W2 - MARGIN_R, STRIDE_R, device="cuda")
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gy, gx = gy.reshape(-1), gx.reshape(-1)
    banks = bank_table(T)
    payload = {"bank_lo": np.array([b[0] for b in banks], np.int64),
               "f0": np.array([b[1] for b in banks], np.int64), "T_global": np.int64(T)}
    t0 = time.time()
    for i, (lo, f0, f1, sa, sb) in enumerate(banks):
        win_t = _resize_win(frames[f0:f1 + 1])
        tl = tuple(range(sa - f0, sb - f0 + 1, 4))
        ey, ex = hot_extra(frames[f0:f1 + 1], sx, sy, gh, gw)
        t2, vi, q = cow_track_bank(model, win_t, tl, torch.cat([gy, ey]), torch.cat([gx, ex]), sx, sy)
        del win_t
        vi = (vi > vis_th).astype(np.float32)
        kp = dedupe_tracks(t2, vi, q)
        t2, vi = t2[:, kp], vi[:, kp]
        t2_t, vi_t = torch.as_tensor(t2, device="cuda"), torch.as_tensor(vi, device="cuda")
        v3 = lift_tracks(t2_t, vi_t, dep[f0:f1 + 1], kee[f0:f1 + 1] if kee is not None else None, K_t)
        v16 = v3[:, None].half().cpu().numpy()
        lifted = ~np.isnan(v16[:, 0, :, 2])
        rt = roundtrip(torch.as_tensor(v16[:, 0].astype(np.float32), device="cuda"), t2_t, K_t)
        ok = rt["p50"] < GATE_P50 and rt["p95"] <= GATE_P95 and rt["n"] > 0
        payload[f"v_{i:02d}"] = v16
        payload[f"conf_{i:02d}"] = lifted.mean(1, keepdims=True).astype(np.float32)
        print(f"[track] bank {i:02d} lo={lo} frames [{f0},{f1}] N={t2.shape[1]} lifted {lifted.mean():.2f} "
              f"reprojection p50={rt['p50']:.3f}px p95={rt['p95']:.3f}px {'ok' if ok else 'WARN'}", flush=True)
        torch.cuda.empty_cache()
    payload["meta"] = json.dumps(dict(tool="cowtracker", res=f"{H2}x{W2}", stride=STRIDE_R, vis_th=vis_th,
                                      seconds=round(time.time() - t0, 1)))
    np.savez_compressed(out, **payload)
    print(f"[track] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--vis_th", type=float, default=VIS_TH)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(case_dir(a.case), a.force, a.vis_th)


if __name__ == "__main__":
    main()
