"""Build a video-editing package from an edited first frame and per-frame edit masks.

  python -m wiw.edit --case bear --name polar --frame0 bear_polar.png --mask masks/ \
      --prompt "A polar bear walks across a rocky enclosure ..."

frame0 : the first source frame edited with any image editor (same framing; 832x464 or larger)
mask   : where the edit lives in every frame - a directory of per-frame PNGs (0/255), an .npz
         with `mask` (T,H,W), a single PNG (used for all frames, static scenes) or `auto`
         (difference between the edited and the original first frame, static)
The edited pixels are composited into frame 0 inside the (dilated) mask, the edited clip is
VAE-encoded, and the masks tell the generator which source pixels must not be trusted. The
edited prompt is T5-encoded as context_edit_<name>.pt. Generate with
  python -m wiw.generate --case bear --traj static --edit polar [--edit_paste 0.5] [--edit_recam]
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from .config import H, W, case_dir
from .prep import encode_prompts, write_mp4


def kernel(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r) + 1, 2 * int(r) + 1))


def dilate(m, r):
    return cv2.dilate(m.astype(np.uint8), kernel(r)) > 0 if r > 0 else m.astype(bool)


def fill_small(m, min_frac=5e-4):
    """Drop small components and fill interior holes."""
    m = m.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_frac * m.size:
            keep[lab == i] = 1
    inv = (1 - keep).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n):
        x, y, w, h = st[i, :4]
        if x > 0 and y > 0 and x + w < W and y + h < H and st[i, cv2.CC_STAT_AREA] < 0.02 * m.size:
            keep[lab == i] = 1
    return keep > 0


def diff_mask(edit, orig, thr=28):
    d = np.abs(edit.astype(np.int16) - orig.astype(np.int16)).max(2).astype(np.uint8)
    d = cv2.GaussianBlur(d, (5, 5), 0)
    return cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_CLOSE, kernel(3)) > 0


def load_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img


def load_masks(spec, T, edit0, orig0):
    if spec == "auto":
        return np.repeat(diff_mask(edit0, orig0)[None], T, 0)
    if spec.endswith(".npz"):
        m = np.load(spec)["mask"]
        m = m > 128 if m.dtype != bool else m
    elif os.path.isdir(spec):
        files = sorted(glob.glob(os.path.join(spec, "*.png")) + glob.glob(os.path.join(spec, "*.jpg")))
        assert files, f"no masks in {spec}"
        m = np.stack([cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 128 for f in files])
    else:
        m = np.repeat((cv2.imread(spec, cv2.IMREAD_GRAYSCALE) > 128)[None], T, 0)
    if m.shape[1:] != (H, W):
        m = np.stack([cv2.resize(x.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0 for x in m])
    if len(m) < T:
        m = np.concatenate([m, np.repeat(m[-1:], T - len(m), 0)])
    return m[:T]


def bbox_sim(m0, mt):
    """Translation + isotropic scale (2x3) between two masks from their centroids and areas."""
    if not m0.any() or not mt.any():
        return None
    y0, x0 = np.where(m0)
    y1, x1 = np.where(mt)
    s = float(np.clip(np.sqrt(mt.sum() / max(m0.sum(), 1)), 0.5, 2.0))
    return np.array([[s, 0, x1.mean() - s * x0.mean()], [0, s, y1.mean() - s * y0.mean()]], np.float32)


def build(case, name, frame0_path, mask_spec, prompt, dil=4, force=False):
    import torch
    out = os.path.join(case, f"edit_{name}.npz")
    if os.path.exists(out) and not force:
        print(f"[edit] {out} exists, skip")
        return
    fr = np.load(os.path.join(case, "frames.npy"))
    T = fr.shape[0]
    orig0 = fr[0]
    edit0 = load_image(frame0_path)
    masks = load_masks(mask_spec, T, edit0, orig0)
    masks = np.stack([fill_small(dilate(m, dil), 1e-4) for m in masks])
    mask0 = masks[0]
    assert mask0.mean() > 1e-4, "empty edit region in frame 0"
    # composite: edited pixels inside the mask (feathered inwards), source pixels elsewhere
    alpha = cv2.GaussianBlur(mask0.astype(np.float32), (7, 7), 0)[..., None]
    alpha = np.where(mask0[..., None], np.maximum(alpha, 0.5), 0.0)
    frame0 = (alpha * edit0 + (1 - alpha) * orig0).round().clip(0, 255).astype(np.uint8)
    frame0[~mask0] = orig0[~mask0]
    # paste material: edited pixels moved with the mask (translation + scale) for --edit_paste
    paste = np.zeros((T, H, W, 3), np.uint8)
    pmask = np.zeros((T, H, W), bool)
    for t in range(T):
        M = np.array([[1, 0, 0], [0, 1, 0]], np.float32) if t == 0 else bbox_sim(mask0, masks[t])
        if M is None:
            continue
        pm = cv2.warpAffine(mask0.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
        pm &= masks[t]
        if pm.any():
            paste[t][pm] = cv2.warpAffine(frame0, M, (W, H), flags=cv2.INTER_LINEAR)[pm]
            pmask[t] = pm
    # latents of the edited clip (frame 0 replaced)
    from .config import CKPT, LINGBOT_REPO
    import sys
    sys.path.insert(0, LINGBOT_REPO)
    from wan.modules.vae2_1 import Wan2_1_VAE
    frames_e = fr.copy()
    frames_e[0] = frame0
    vae = Wan2_1_VAE(vae_pth=os.path.join(CKPT, "Wan2.1_VAE.pth"), device="cuda")
    vid = torch.from_numpy(frames_e).float().permute(3, 0, 1, 2) / 127.5 - 1.0
    with torch.no_grad():
        lat = vae.encode([vid.cuda()])[0].cpu().numpy()
    del vae
    torch.cuda.empty_cache()
    encode_prompts(case, {f"edit_{name}": prompt}, force=force)
    cov = masks.reshape(T, -1).mean(1)
    np.savez_compressed(out, frame0=frame0, frame0_orig=orig0, latents=lat, mask=(masks * 255).astype(np.uint8),
                        paste=paste, pmask=(pmask * 255).astype(np.uint8), prompt=prompt,
                        meta=json.dumps(dict(frame0=os.path.abspath(frame0_path), mask=mask_spec, dilate=dil)))
    # preview: source | edit mask | frame with edit region cut
    prev = []
    for t in range(0, T, max(1, T // 48)):
        ov = fr[t].copy()
        ov[masks[t]] = (0.45 * ov[masks[t]] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
        cut = fr[t].copy()
        cut[masks[t]] = 128
        prev.append(np.concatenate([frame0 if t == 0 else fr[t], ov, cut], 1))
    write_mp4(prev, os.path.join(case, f"edit_{name}_preview.mp4"), fps=8, crf=22)
    print(f"[edit] {out}: coverage frame0 {100 * cov[0]:.1f}%  median {100 * np.median(cov):.1f}%  "
          f"preview edit_{name}_preview.mp4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--frame0", required=True, help="edited first frame (image)")
    ap.add_argument("--mask", required=True, help="per-frame masks: dir | .npz | single png | auto")
    ap.add_argument("--prompt", required=True, help="prompt describing the edited video")
    ap.add_argument("--dilate", type=int, default=4, help="mask dilation radius in pixels")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(case_dir(a.case), a.name, a.frame0, a.mask, a.prompt, a.dilate, a.force)


if __name__ == "__main__":
    main()
