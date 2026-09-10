"""Video depth for a case with Video-Depth-Anything (ViT-L).

  python -m wiw.depth --case tennis

Writes <case>/depth.npz with `disp` (T,464,832) float16: relative inverse depth, min-max
normalised over the whole clip to [0,1] (1 = nearest). Requires the Video-Depth-Anything
repository (WIW_VDA_REPO) and its ViT-L checkpoint (WIW_VDA_CKPT).
"""
import argparse
import os
import sys

import numpy as np

from .config import VDA_REPO, VDA_CKPT, FPS, case_dir


def estimate(frames_rgb, fp32=False, input_size=518):
    import torch
    sys.path.insert(0, VDA_REPO)
    from video_depth_anything.video_depth import VideoDepthAnything
    model = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(VDA_CKPT, map_location="cpu"), strict=True)
    model = model.to("cuda").eval()
    depths, _ = model.infer_video_depth(frames_rgb, FPS, input_size=input_size, device="cuda", fp32=fp32)
    return np.asarray(depths, np.float32)


def run(case, force=False):
    out = os.path.join(case, "depth.npz")
    if os.path.exists(out) and not force:
        print(f"[depth] {out} exists, skip")
        return
    frames = np.load(os.path.join(case, "frames.npy"))
    raw = estimate(frames)
    assert raw.shape[0] == frames.shape[0], (raw.shape, frames.shape)
    lo, hi = float(raw.min()), float(raw.max())
    disp = ((raw - lo) / (hi - lo)).astype(np.float16)
    np.savez_compressed(out, disp=disp, meta="Video-Depth-Anything vitl; inverse depth, min-max normalised, 1 = near")
    print(f"[depth] {out} {disp.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(case_dir(a.case), a.force)


if __name__ == "__main__":
    main()
