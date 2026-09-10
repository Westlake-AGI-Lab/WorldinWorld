"""Run one of the bundled examples.

  python tools/run_example.py examples/tennis_bullet_time
  python tools/run_example.py examples/robot_bedroom_rotation

Each example ships its source video plus depth, prompt embeddings and trajectory under
<example>/data/; frames and VAE latents are recomputed from the video (about a minute), so only the
base model is needed. Pass --from_scratch to run every step yourself (prep -> depth ->
[bullet time] -> trajectory -> generate), --cgar to also compute point tracks and enable
correspondence-guided attention (slow, needs CoWTracker).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wiw.config import case_dir  # noqa: E402

PY = sys.executable


def sh(*args):
    cmd = [PY, "-m"] + [str(a) for a in args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("example", help="example directory containing config.json")
    ap.add_argument("--from_scratch", action="store_true", help="ignore the bundled data and run every step")
    ap.add_argument("--cgar", action="store_true", help="compute point tracks and enable correspondence-guided attention")
    ap.add_argument("--skip_generate", action="store_true")
    ap.add_argument("--generate_args", default="", help='extra arguments for wiw.generate, e.g. --generate_args="--seed 7"')
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.example, "config.json")))
    case = cfg["case"]
    video = os.path.join(a.example, cfg["video"])
    gen_case = cfg["bullet"]["case"] if "bullet" in cfg else case
    t = cfg["traj"]

    if not a.from_scratch:
        src = os.path.join(a.example, "data", gen_case)
        if not os.path.exists(os.path.join(case_dir(gen_case), "depth.npz")):
            shutil.copytree(src, case_dir(gen_case), dirs_exist_ok=True)
            print(f"copied bundled data -> {case_dir(gen_case)}")
        # the prompt embeddings are shared between a clip and its bullet-time derivative
        os.makedirs(case_dir(case), exist_ok=True)
        for f in os.listdir(case_dir(gen_case)):
            if f.startswith(("context", "prompt")) and not os.path.exists(os.path.join(case_dir(case), f)):
                shutil.copy(os.path.join(case_dir(gen_case), f), case_dir(case))
    # frames + VAE latents (prompt embeddings are only encoded when missing)
    prep = ["wiw.prep", "--video", video, "--case", case, "--prompt", cfg["prompt"]]
    if cfg.get("scene_prompt"):
        prep += ["--scene_prompt", cfg["scene_prompt"]]
    for k, v in cfg.get("prep", {}).items():
        prep += [f"--{k}"] + ([] if v is True else [str(v)])
    sh(*prep)
    if a.from_scratch:
        sh("wiw.depth", "--case", case)
    if "bullet" in cfg and not os.path.exists(os.path.join(case_dir(gen_case), "frames.npy")):
        sh("wiw.bullet", "--case", case, "--out", gen_case, "--freeze", cfg["bullet"]["freeze"])
    if a.cgar and not os.path.exists(os.path.join(case_dir(gen_case), "tracks.npz")):
        sh("wiw.track", "--case", gen_case)
    if not os.path.exists(os.path.join(case_dir(gen_case), f"traj_{t['name']}", "poses.npy")):
        traj = ["wiw.traj", "--case", gen_case, "--name", t["name"], "--pivot", t.get("pivot", "auto:0")]
        for w in t.get("window", []):
            traj += ["--window", w]
        for k in ("motion", "yaw_plan", "explore"):
            if k in t:
                traj += [f"--{k}", t[k]]
        sh(*traj)
    if a.skip_generate:
        return
    gen = ["wiw.generate", "--case", gen_case, "--traj", t["name"]]
    for k, v in cfg.get("generate", {}).items():
        gen += [f"--{k}"] + ([] if v is True else [str(v)])
    if a.cgar:
        gen.append("--cgar")
    gen += a.generate_args.split()
    sh(*gen)


if __name__ == "__main__":
    main()
