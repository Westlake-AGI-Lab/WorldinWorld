"""Download the model weights used by World in World.

  python tools/download_weights.py                 # everything into ./checkpoints
  python tools/download_weights.py --only lingbot  # base video model only (~86 GB)

Weights:
  lingbot   robbyant/lingbot-world-v2-14b-causal-fast  (~86 GB, CC BY-NC-SA 4.0)
  vda       depth-anything/Video-Depth-Anything-Large   (~1.5 GB, CC BY-NC 4.0)
  cow       facebook/cowtracker                          (~3.9 GB, see repository licence)
Downloads resume when re-run.
"""
import argparse
import os

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--only", default="lingbot,vda,cow")
    ap.add_argument("--endpoint", default=None, help="alternative Hugging Face endpoint URL")
    ap.add_argument("--vda_repo", default=os.path.join(ROOT, "third_party", "Video-Depth-Anything"))
    a = ap.parse_args()
    if a.endpoint:
        os.environ["HF_ENDPOINT"] = a.endpoint
    want = set(a.only.split(","))
    os.makedirs(a.dst, exist_ok=True)
    if "lingbot" in want:
        d = os.path.join(a.dst, "lingbot-world-v2-14b-causal-fast")
        snapshot_download("robbyant/lingbot-world-v2-14b-causal-fast", local_dir=d)
        for f in ["Wan2.1_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/tokenizer.json",
                  "transformers/diffusion_pytorch_model.safetensors.index.json"] + \
                 [f"transformers/model-0000{i}-of-00008.safetensors" for i in range(1, 9)]:
            assert os.path.exists(os.path.join(d, f)), f"missing {f}"
        print(f"[ok] LingBot-World-v2 -> {d}")
    if "vda" in want:
        d = os.path.join(a.vda_repo, "checkpoints")
        os.makedirs(d, exist_ok=True)
        hf_hub_download("depth-anything/Video-Depth-Anything-Large", "video_depth_anything_vitl.pth", local_dir=d)
        print(f"[ok] Video-Depth-Anything ViT-L -> {d}")
    if "cow" in want:
        d = os.path.join(a.dst, "cowtracker")
        snapshot_download("facebook/cowtracker", local_dir=d)
        print(f"[ok] CoWTracker -> {d}")


if __name__ == "__main__":
    main()
