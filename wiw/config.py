"""Paths and constants shared by all World-in-World tools.

Override locations with environment variables:
  WIW_WORKSPACE   directory holding one sub-directory per case      (default ./workspace)
  WIW_CKPT        LingBot-World-v2 14B causal-fast checkpoint dir   (default ./checkpoints/lingbot-world-v2-14b-causal-fast)
  WIW_VDA_REPO    Video-Depth-Anything repository                   (default ./third_party/Video-Depth-Anything)
  WIW_VDA_CKPT    video_depth_anything_vitl.pth                     (default <repo>/checkpoints/video_depth_anything_vitl.pth)
  WIW_COW_REPO    CoWTracker repository                             (default ./third_party/CoWTracker)
  WIW_COW_CKPT    cowtracker_model.pth                              (default ./checkpoints/cowtracker/cowtracker_model.pth)
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINGBOT_REPO = os.path.join(ROOT, "lingbot_world")

WORKSPACE = os.environ.get("WIW_WORKSPACE", os.path.join(ROOT, "workspace"))
CKPT = os.environ.get("WIW_CKPT", os.path.join(ROOT, "checkpoints", "lingbot-world-v2-14b-causal-fast"))
VDA_REPO = os.environ.get("WIW_VDA_REPO", os.path.join(ROOT, "third_party", "Video-Depth-Anything"))
VDA_CKPT = os.environ.get("WIW_VDA_CKPT", os.path.join(VDA_REPO, "checkpoints", "video_depth_anything_vitl.pth"))
COW_REPO = os.environ.get("WIW_COW_REPO", os.path.join(ROOT, "third_party", "CoWTracker"))
COW_CKPT = os.environ.get("WIW_COW_CKPT", os.path.join(ROOT, "checkpoints", "cowtracker", "cowtracker_model.pth"))

# Working resolution of the video model and the intrinsics used for every case.
# K4 = (fx, fy, cx, cy) is stored at the 832x480 reference; frames are 832x464 (fy, cy scaled).
H, W = 464, 832
K4 = np.array([415.5298, 415.6922, 415.77786, 239.77779], np.float32)
FPS = 16

# Causal video model layout: 4 latents per chunk, 6 sink + 8 rolling-window + 4 current slots.
CHUNK = 4
SINK_LAT = 6
WINDOW_LAT = 18
# Trajectories repeat their last pose for the final DWELL frames.
DWELL = 4


def case_dir(name_or_path):
    if os.path.isdir(name_or_path):
        return os.path.abspath(name_or_path)
    return os.path.join(WORKSPACE, name_or_path)


def frames_to_latents(t):
    return (t - 1) // 4 + 1


def valid_frame_count(n):
    """Largest T <= n with T = 16m + 5 (the model generates T - 8 frames from such a clip)."""
    return n - (n - 5) % 16
