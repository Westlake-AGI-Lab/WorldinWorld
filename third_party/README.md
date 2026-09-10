# Third-party repositories

Video-Depth-Anything is needed to process your own videos (the bundled examples ship their depth);
CoWTracker only for the optional correspondence-guided attention (`--cgar`). Clone them here, or point `WIW_VDA_REPO` / `WIW_COW_REPO` at existing checkouts; they are used as plain
Python sources and need nothing beyond `requirements.txt`.

```bash
cd third_party
git clone https://github.com/DepthAnything/Video-Depth-Anything.git
git clone --recurse-submodules https://github.com/facebookresearch/cowtracker.git CoWTracker
```

Video-Depth-Anything provides the per-frame depth of the source video (`wiw.depth`); its ViT-L weights
(`video_depth_anything_vitl.pth`, CC BY-NC 4.0) are fetched by `python tools/download_weights.py --only vda`.
CoWTracker provides the dense point tracks for correspondence-guided attention (`wiw.track`); its
weights (`facebook/cowtracker` on Hugging Face, FAIR Noncommercial Research License) are fetched by
`python tools/download_weights.py --only cow`.
