"""Prepare a case directory from a source video.

  python -m wiw.prep --video clip.mp4 --case tennis \
      --prompt "A yellow tennis ball bounces on a grass court ..." \
      --scene_prompt "A green grass court ... everything perfectly still, only the camera moves"

Writes <workspace>/<case>/{frames.npy, source.mp4, meta.json, prompt*.txt, context*.pt,
latents.pt, traj_static/}. Frames are cover-resized and centre-cropped to 832x464 and the
clip is trimmed to T = 16m + 5 frames (the model generates T - 8 frames).
"""
import argparse
import json
import os

import cv2
import numpy as np

from .config import H, W, K4, FPS, CKPT, LINGBOT_REPO, case_dir

MAX_PAD = 8


def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    assert frames, f"no frames read from {path}"
    return frames, fps


def to_working_res(f):
    h, w = f.shape[:2]
    s = max(W / w, H / h)
    rw, rh = int(round(w * s)), int(round(h * s))
    g = cv2.resize(f, (rw, rh), interpolation=cv2.INTER_AREA)
    top, left = (rh - H) // 2, (rw - W) // 2
    return cv2.cvtColor(g[top:top + H, left:left + W], cv2.COLOR_BGR2RGB), \
        dict(src_hw=[h, w], scale=s, crop_top=top, crop_left=left)


def dedup_indices(frames, rel=0.15):
    """Drop near-duplicate consecutive frames (3:2 pulldown), threshold relative to the median."""
    g = [cv2.cvtColor(cv2.resize(f, (208, 116), interpolation=cv2.INTER_AREA),
                      cv2.COLOR_BGR2GRAY).astype(np.int16) for f in frames]
    d = np.array([np.abs(g[i] - g[i - 1]).mean() for i in range(1, len(g))])
    thr = rel * float(np.median(d))
    dup = set((np.where(d < thr)[0] + 1).tolist())
    return [i for i in range(len(frames)) if i not in dup], len(dup)


def target_len(n):
    """Nearest T = 16m + 5: pad up to MAX_PAD frames (repeat last), otherwise trim."""
    up = n if (n - 5) % 16 == 0 else n + (16 - (n - 5) % 16)
    if up - n <= MAX_PAD:
        return up, up - n, 0
    down = up - 16
    assert down >= 21, f"clip too short ({n} frames)"
    return down, 0, n - down


def write_mp4(frames_rgb, path, fps=FPS, crf=18):
    import imageio
    wr = imageio.get_writer(path, fps=fps, codec="libx264",
                            ffmpeg_params=["-crf", str(crf), "-pix_fmt", "yuv420p"])
    for f in frames_rgb:
        wr.append_data(f)
    wr.close()


def write_static_traj(case, T):
    d = os.path.join(case, "traj_static")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "poses.npy"), np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)))
    np.save(os.path.join(d, "intrinsics.npy"), np.tile(K4, (T, 1)))


def prep_frames(video, case, start=0, max_frames=None, target_fps=None, dedup=True, force=False):
    fp = os.path.join(case, "frames.npy")
    if os.path.exists(fp) and not force:
        print(f"[prep] {fp} exists, skip (use --force to redo)")
        return json.load(open(os.path.join(case, "meta.json")))
    allf, fps = read_video(video)
    M = len(allf)
    idx = np.arange(M)
    n_dup = 0
    if dedup:
        keep, n_dup = dedup_indices(allf)
        idx = np.asarray(keep)
    if target_fps and fps > target_fps + 0.5:
        n_rs = int(round(len(idx) * target_fps / fps))
        idx = idx[np.round(np.linspace(0, len(idx) - 1, n_rs)).astype(int)]
    idx = idx[start:]
    if max_frames:
        idx = idx[:max_frames]
    T, pad, drop = target_len(len(idx))
    idx = idx[:len(idx) - drop] if drop else idx
    frames, tf = [], None
    for k in idx:
        g, tf = to_working_res(allf[k])
        frames.append(g)
    frames = np.stack(frames)
    if pad:
        frames = np.concatenate([frames, np.repeat(frames[-1:], pad, 0)], 0)
    assert frames.shape[0] == T and (T - 5) % 16 == 0
    os.makedirs(case, exist_ok=True)
    np.save(fp, frames)
    write_mp4(frames, os.path.join(case, "source.mp4"))
    write_static_traj(case, T)
    meta = dict(video=os.path.abspath(video), src_frames=M, src_fps=fps, dedup_removed=n_dup,
                start=start, T=int(T), pad=int(pad), drop=int(drop), n_traj=int(T - 8),
                sample_idx=idx.tolist(), transform=tf)
    json.dump(meta, open(os.path.join(case, "meta.json"), "w"), indent=1)
    print(f"[prep] {M} frames @ {fps:.2f} fps -> T={T} (pad {pad}, drop {drop}); "
          f"{T - 8} frames will be generated ({(T - 8) / FPS:.1f} s at {FPS} fps)")
    return meta


def encode_latents(case, force=False):
    import sys
    import torch
    sys.path.insert(0, LINGBOT_REPO)
    from wan.modules.vae2_1 import Wan2_1_VAE
    out = os.path.join(case, "latents.pt")
    if os.path.exists(out) and not force:
        return
    frames = np.load(os.path.join(case, "frames.npy"))
    vae = Wan2_1_VAE(vae_pth=os.path.join(CKPT, "Wan2.1_VAE.pth"), device="cuda")
    vid = torch.from_numpy(frames).float().permute(3, 0, 1, 2) / 127.5 - 1.0
    with torch.no_grad():
        lat = vae.encode([vid.cuda()])[0].cpu()
    assert lat.shape[1] == (frames.shape[0] - 1) // 4 + 1
    torch.save(lat, out)
    print(f"[prep] latents {tuple(lat.shape)}")
    del vae
    torch.cuda.empty_cache()


def encode_prompts(case, prompts, force=False):
    """prompts: {name: text}; name '' is the main prompt -> context.pt / prompt.txt."""
    import sys
    import torch
    sys.path.insert(0, LINGBOT_REPO)
    from wan.modules.t5 import T5EncoderModel
    todo = {n: t for n, t in prompts.items()
            if force or not os.path.exists(os.path.join(case, f"context{'_' + n if n else ''}.pt"))}
    if not todo:
        return
    t5 = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=torch.device("cuda"),
                        checkpoint_path=os.path.join(CKPT, "models_t5_umt5-xxl-enc-bf16.pth"),
                        tokenizer_path=os.path.join(CKPT, "google/umt5-xxl"))
    for n, txt in todo.items():
        sfx = f"_{n}" if n else ""
        with torch.no_grad():
            ctx = t5([txt], torch.device("cuda"))[0].cpu()
        torch.save(ctx, os.path.join(case, f"context{sfx}.pt"))
        with open(os.path.join(case, f"prompt{sfx}.txt"), "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"[prep] context{sfx}.pt  <- {txt[:70]}...")
    del t5
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description="Prepare a case from a source video.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--case", required=True, help="case name (under WIW_WORKSPACE) or directory")
    ap.add_argument("--prompt", default=None, help="full prompt for the first chunk (may mention the subject)")
    ap.add_argument("--scene_prompt", default=None,
                    help="scene-only prompt for later chunks (no subjects); saved as context_scene.pt")
    ap.add_argument("--extra_prompt", action="append", default=[],
                    help="additional prompt 'name=text' -> context_<name>.pt (e.g. a frozen-time prompt)")
    ap.add_argument("--start", type=int, default=0, help="first source frame to keep")
    ap.add_argument("--max_frames", type=int, default=None, help="keep at most this many frames")
    ap.add_argument("--target_fps", type=float, default=None,
                    help="resample the source to this fps before trimming (playback is always 16 fps)")
    ap.add_argument("--no_dedup", action="store_true", help="keep pulldown duplicate frames")
    ap.add_argument("--skip_encode", action="store_true", help="only write frames (no GPU needed)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    case = case_dir(a.case)
    prep_frames(a.video, case, a.start, a.max_frames, a.target_fps, not a.no_dedup, a.force)
    prompts = {}
    if a.prompt:
        prompts[""] = a.prompt
    if a.scene_prompt:
        prompts["scene"] = a.scene_prompt
    for e in a.extra_prompt:
        n, _, t = e.partition("=")
        prompts[n] = t
    if a.skip_encode:
        for n, t in prompts.items():
            with open(os.path.join(case, f"prompt{'_' + n if n else ''}.txt"), "w", encoding="utf-8") as f:
                f.write(t)
        return
    encode_latents(case, a.force)
    encode_prompts(case, prompts, a.force)


if __name__ == "__main__":
    main()
