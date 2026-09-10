"""Derive a bullet-time case: freeze the source at given frames for a number of frames.

  python -m wiw.bullet --case tennis --out tennis_bt --freeze 45:80

`--freeze f:dur[,f2:dur2]` repeats source frame f for dur frames (dur must be a multiple of
16 so the clip stays 16m + 5). Frames, latents, depth, source camera and prompts are re-indexed;
tracks (wiw.track) must be recomputed on the derived case. The camera motion is then placed
inside the frozen window with `wiw.traj --window lo-hi:motion`.
"""
import argparse
import json
import os
import shutil

import numpy as np

from .config import case_dir
from .prep import write_static_traj, write_mp4


def remap_indices(freezes, T):
    idx, cur = [], 0
    for f, dur in freezes:
        assert cur <= f < T, (f, cur, T)
        idx += list(range(cur, f)) + [f] * dur
        cur = f
    idx += list(range(cur, T))
    return np.asarray(idx, np.int64)


def derive(base, out, freezes):
    """Re-index every artefact that exists in `base` (frames, latents, depth, source camera,
    prompts); missing ones are skipped so a frames-only base works too."""
    import torch
    fr = np.load(os.path.join(base, "frames.npy"), mmap_mode="r")
    T = fr.shape[0]
    idx = remap_indices(freezes, T)
    T_new = len(idx)
    assert (T_new - 5) % 16 == 0, f"T_new={T_new} is not 16m+5 (freeze durations must sum to a multiple of 16)"
    os.makedirs(out, exist_ok=True)
    frames = np.asarray(fr[idx])
    np.save(os.path.join(out, "frames.npy"), frames)
    write_mp4(frames, os.path.join(out, "source.mp4"))
    if os.path.exists(os.path.join(base, "latents.pt")):
        lat = torch.load(os.path.join(base, "latents.pt"))
        n_lat_new = (T_new + 3) // 4
        lat_idx = np.minimum(idx[:n_lat_new] // 4, lat.shape[1] - 1)
        torch.save(lat[:, lat_idx], os.path.join(out, "latents.pt"))
    if os.path.exists(os.path.join(base, "depth.npz")):
        d = np.load(os.path.join(base, "depth.npz"))
        np.savez_compressed(os.path.join(out, "depth.npz"),
                            **{k: (d[k][idx] if d[k].ndim == 3 else d[k]) for k in d.files})
    cam = os.path.join(base, "srccam.npz")
    if os.path.exists(cam):
        z = np.load(cam)
        payload = {k: z[k] for k in z.files}
        payload["c2w"] = z["c2w"][np.minimum(idx[:T_new - 8], len(z["c2w"]) - 1)]
        np.savez_compressed(os.path.join(out, "srccam.npz"), **payload)
    for f in os.listdir(base):
        if f.startswith(("context", "prompt")) and f.endswith((".pt", ".txt")):
            shutil.copy(os.path.join(base, f), os.path.join(out, f))
    write_static_traj(out, T_new)
    wins, off = [], 0
    for f, dur in freezes:
        wins.append([int(f + off), int(f + off + dur - 1)])
        off += dur
    meta = json.load(open(os.path.join(base, "meta.json"))) if os.path.exists(os.path.join(base, "meta.json")) else {}
    meta.update(base_case=os.path.abspath(base), T=int(T_new), n_traj=int(T_new - 8),
                freezes=[[int(f), int(d_)] for f, d_ in freezes], frozen_windows=wins, remap_idx=idx.tolist())
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"[bullet] {base} -> {out}: T {T} -> {T_new}; frozen windows (trajectory frames) {wins}")
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="base case")
    ap.add_argument("--out", required=True, help="derived case name")
    ap.add_argument("--freeze", required=True, help="f:dur[,f2:dur2] in source frame indices")
    a = ap.parse_args()
    fr = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in a.freeze.split(",")]
    derive(case_dir(a.case), case_dir(a.out), fr)


if __name__ == "__main__":
    main()
