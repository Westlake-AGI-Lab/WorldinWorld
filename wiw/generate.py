"""Re-cinematography / bullet-time / editing generation for a prepared case.

  python -m wiw.generate --case tennis_bt --traj arc75 --prompt_schedule 3-7:scene
  python -m wiw.generate --case robot_bedroom --traj pantilt --prompt scene
  python -m wiw.generate --case bear --traj static --edit polar

Needs (see README): <case>/{frames.npy, latents.pt, context*.pt, depth.npz, traj_<name>/}
(+ tracks.npz when --cgar is given).
Outputs <out>/<tag>.mp4 plus <out>/<tag>/ with the warped evidence (_input.mp4), the token
weights (_mask.mp4), their product (_maskfused.mp4) and _meta.json.
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

from .config import FPS, case_dir
from .model import load_pipe
from .pseudo_hist import PseudoHistBias
from .runner import Runner

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])


def parse_steps(s):
    return tuple(int(x) for x in s.split(",")) if s else ()


def save_video(video, path, fps=FPS, crf=26, hq_dir=None):
    import imageio
    arr = ((video.clamp(-1, 1).permute(1, 2, 3, 0).cpu().numpy() + 1) * 127.5).round().astype("uint8")

    def write(p, crf_, preset="medium"):
        w = imageio.get_writer(p, fps=fps, codec="libx264",
                               ffmpeg_params=["-crf", str(crf_), "-preset", preset, "-pix_fmt", "yuv420p"])
        for f in arr:
            w.append_data(f)
        w.close()

    if hq_dir is not None:
        import cv2
        os.makedirs(os.path.join(hq_dir, "frames"), exist_ok=True)
        for i, f in enumerate(arr):
            cv2.imwrite(os.path.join(hq_dir, "frames", f"f{i:05d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 98])
        write(os.path.join(hq_dir, "hq.mp4"), 12, "slow")
    write(path, crf)
    return arr


def save_evidence_videos(runner, out_dir, tag, n_frames, fps=FPS):
    import cv2
    import imageio
    evid = runner.evid
    src = runner.frames
    H, W = src.shape[1:3]
    gh, gw = H // 16, W // 16
    ws = {name: imageio.get_writer(os.path.join(out_dir, f"{tag}_{name}.mp4"), fps=fps, codec="libx264",
                                   ffmpeg_params=["-crf", "26", "-preset", "medium", "-pix_fmt", "yuv420p"])
          for name in ("input", "mask", "maskfused")}
    for f in range(n_frames):
        inp = evid["in"].get(f)
        if inp is None:
            inp, wg = src[min(f, len(src) - 1)], np.ones((gh, gw), np.float32)
        else:
            wg = evid["w"].get(f, np.zeros((gh, gw), np.float32))
        u8m = cv2.resize((np.clip(wg, 0, 1) * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        heat = np.ascontiguousarray(cv2.applyColorMap(u8m, cv2.COLORMAP_VIRIDIS)[..., ::-1])
        wp = cv2.resize(np.clip(wg, 0, 1), (W, H), interpolation=cv2.INTER_NEAREST)[..., None]
        fused = np.ascontiguousarray((inp.astype(np.float32) * wp).astype(np.uint8))
        inp2 = np.ascontiguousarray(inp.copy())
        cv2.putText(inp2, f"f{f} warped evidence", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(heat, f"f{f} token weights", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(fused, f"f{f} evidence x weights", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for name, im in (("input", inp2), ("mask", heat), ("maskfused", fused)):
            ws[name].append_data(im)
    for wv in ws.values():
        wv.close()


def load_context(case, name):
    p = os.path.join(case, f"context{'_' + name if name else ''}.pt")
    assert os.path.exists(p), f"missing prompt embedding {p} (encode it with wiw.prep --extra_prompt)"
    return torch.load(p)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default=None, help="case name (under WIW_WORKSPACE) or directory")
    ap.add_argument("--traj", default=None, help="trajectory name: <case>/traj_<name>/")
    ap.add_argument("--out", default=None, help="output directory (default <case>/out)")
    ap.add_argument("--tag", default=None, help="output name (default <case>_<traj>)")
    ap.add_argument("--prompt", default="", help="prompt name used for all chunks ('' = context.pt, "
                    "'scene' = context_scene.pt, ...)")
    ap.add_argument("--prompt_schedule", default=None,
                    help="'lo-hi:name[;lo2-hi2:name2]' - chunks lo..hi (inclusive) use context_<name>.pt")
    ap.add_argument("--edit", default=None, help="edit package name: <case>/edit_<name>.npz (see wiw.edit)")
    ap.add_argument("--edit_paste", type=float, default=None,
                    help="paste the edited pixels into the evidence with this weight (0-1]")
    ap.add_argument("--edit_recam", action="store_true",
                    help="editing + camera motion: cut the edit region from the source before warping")
    ap.add_argument("--srccam", default="auto", choices=["auto", "on", "off"],
                    help="fold <case>/srccam.npz (moving source camera) into the camera embedding")
    ap.add_argument("--cgar", action="store_true",
                    help="enable correspondence-guided attention (needs <case>/tracks.npz from wiw.track)")
    ap.add_argument("--cgar_gamma", type=float, default=2.0)
    ap.add_argument("--cgar_vote_min", type=int, default=4)
    ap.add_argument("--ph_steps", default="0,1,2", help="denoising steps (of 4) on which the evidence is attached")
    ap.add_argument("--ewa_w", type=float, default=0.5, help="evidence-weighted amplification factor (0 = off)")
    ap.add_argument("--ewa_steps", default="0,1,2")
    ap.add_argument("--no_skirt", action="store_true", help="disable mesh-skirt occlusion in the warp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shift", type=float, default=4.0, help="flow-matching timestep shift")
    ap.add_argument("--max_chunks", type=int, default=None)
    ap.add_argument("--save_hq", action="store_true", help="also save q98 jpg frames and a crf12 mp4")
    ap.add_argument("--srcq_gpu", action="store_true", help="keep CGAR source K/V on the GPU (faster, more memory)")
    ap.add_argument("--ckpt", default=None)
    return ap


def run(a, pipe=None):
    """Run one job; returns the loaded pipe so several jobs can share the weights."""
    assert a.case and a.traj, "--case and --traj are required"
    case = case_dir(a.case)
    traj_dir = os.path.join(case, f"traj_{a.traj}")
    assert os.path.isdir(traj_dir), f"missing trajectory {traj_dir} (create it with wiw.traj)"
    out_dir = a.out or os.path.join(case, "out")
    tag = a.tag or f"{os.path.basename(os.path.normpath(case))}_{a.traj}" + (f"_edit_{a.edit}" if a.edit else "")
    os.makedirs(out_dir, exist_ok=True)

    frames = np.load(os.path.join(case, "frames.npy"))
    latents = torch.load(os.path.join(case, "latents.pt"))
    disp = np.load(os.path.join(case, "depth.npz"))["disp"].astype(np.float32)
    prompt = a.prompt
    edit_cut = edit_paste = edit_hole = None
    if a.edit:
        ez = np.load(os.path.join(case, f"edit_{a.edit}.npz"), allow_pickle=True)
        assert tuple(ez["frame0"].shape) == tuple(frames[0].shape), "edit package does not match the case frames"
        assert tuple(ez["latents"].shape) == tuple(latents.shape), "edit package does not match the case latents"
        frames = frames.copy()
        frames[0] = ez["frame0"]
        latents = torch.from_numpy(np.asarray(ez["latents"])).to(latents.dtype)
        mask = np.asarray(ez["mask"]) > 128
        assert mask.shape[1:] == frames.shape[1:3] and len(mask) >= len(frames) - 8, "edit masks do not match the frames"
        if not a.prompt:
            prompt = f"edit_{a.edit}"
        if a.edit_recam:
            assert a.edit_paste is None, "--edit_recam does not support --edit_paste"
            edit_hole = mask
        else:
            edit_cut = mask
        if a.edit_paste is not None:
            assert 0.0 < a.edit_paste <= 1.0
            edit_paste = dict(rgb=np.asarray(ez["paste"]), mask=np.asarray(ez["pmask"]) > 128, w=float(a.edit_paste))
        logging.info(f"[edit] {a.edit}: mask coverage {100 * mask.reshape(len(mask), -1).mean(1).mean():.1f}%"
                     + (" | recam (source holes)" if a.edit_recam else "")
                     + (f" | paste w={a.edit_paste}" if a.edit_paste is not None else ""))
    contexts = [load_context(case, prompt)]
    srccam = os.path.join(case, "srccam.npz")
    if a.srccam == "on":
        assert os.path.exists(srccam), f"--srccam on but {srccam} is missing"
    srccam = srccam if (a.srccam == "on" or (a.srccam == "auto" and os.path.exists(srccam))) else None
    if srccam:
        logging.info(f"[camera] using source camera poses {srccam}")
    tracks = None
    if a.cgar:
        tracks = os.path.join(case, "tracks.npz")
        assert os.path.exists(tracks), f"--cgar needs {tracks}: run  python -m wiw.track --case {os.path.basename(os.path.normpath(case))}"

    if pipe is None:
        pipe = load_pipe(a.ckpt)
    else:
        pipe.model.to(pipe.device)
        PseudoHistBias.detach(pipe)
        torch.cuda.empty_cache()
    runner = Runner(pipe, frames, latents, traj_dir, contexts, seed=a.seed, shift=a.shift, srccam_npz=srccam)
    chunk_ctx_ids = [0] * runner.n_chunks
    if a.prompt_schedule:
        for grp in a.prompt_schedule.split(";"):
            rng, name = grp.split(":")
            runner.contexts.append(load_context(case, name).to(runner.device))
            kid = len(runner.contexts) - 1
            for r in rng.split(","):
                lo_c, hi_c = (int(x) for x in r.split("-"))
                assert 0 <= lo_c <= hi_c < runner.n_chunks, f"chunk range {r} outside 0..{runner.n_chunks - 1}"
                for c in range(lo_c, hi_c + 1):
                    chunk_ctx_ids[c] = kid
        logging.info(f"prompt schedule per chunk: {chunk_ctx_ids}")
    runner.edit_cut, runner.edit_paste = edit_cut, edit_paste
    runner.setup_warp(disp, tracks_npz=tracks, skirt=not a.no_skirt, srcq_gamma=a.cgar_gamma,
                      vote_min=a.cgar_vote_min, edit_hole_seq=edit_hole)
    dsg = dict(w_warp=a.ewa_w, steps=parse_steps(a.ewa_steps), skip_last=True, tau=1.0) if a.ewa_w > 0 else None
    ph_bank = PseudoHistBias(num_layers=pipe.model.num_layers, dsg=dsg)
    ph_bank.attach(pipe)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    video = runner.generate(ph_bank, ph_steps=parse_steps(a.ph_steps), chunk_ctx_ids=chunk_ctx_ids,
                            max_chunks=a.max_chunks, srcq_cpu=not a.srcq_gpu)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    ev_dir = os.path.join(out_dir, tag)
    os.makedirs(ev_dir, exist_ok=True)
    mp4 = os.path.join(out_dir, f"{tag}.mp4")
    save_video(video, mp4, hq_dir=os.path.join(out_dir, f"{tag}_hq") if a.save_hq else None)
    save_evidence_videos(runner, ev_dir, tag, video.shape[1])
    meta = dict(tag=tag, case=case, traj=a.traj, args=vars(a), n_frames=int(video.shape[1]),
                gen_seconds=round(dt, 1), peak_gpu_gb=round(peak, 2), log=runner.log)
    json.dump(meta, open(os.path.join(ev_dir, f"{tag}_meta.json"), "w"), indent=1, default=str)
    logging.info(f"done in {dt:.0f}s, peak GPU {peak:.1f} GB -> {mp4}")
    PseudoHistBias.detach(pipe)
    return pipe


def main():
    ap = build_parser()
    ap.add_argument("--jobs", default=None,
                    help="text file with one argument line per job; the model is loaded once for all jobs")
    a = ap.parse_args()
    if not a.jobs:
        run(a)
        return
    import shlex
    pipe = None
    for line in open(a.jobs):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        job = build_parser().parse_args(shlex.split(line))
        job.ckpt = job.ckpt or a.ckpt
        logging.info(f"=== job: {line}")
        pipe = run(job, pipe)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
