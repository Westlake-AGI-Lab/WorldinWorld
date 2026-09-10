# Tennis — bullet time

`tennis.mp4` is a static close-up of a bouncing ball (1280x720, 24 fps, 6.6 s). The ball is frozen at
frame 45 (its highest point) for 80 frames while the camera sweeps an arc from -75° to +75° around it
and comes back; then the bounce continues.

```bash
python tools/run_example.py examples/tennis_bullet_time
```

`data/tennis_bt/` holds the depth, prompt embeddings and the `arc75` trajectory of the frozen clip, so only
the base model is needed. Output:
`workspace/tennis_bt/out/tennis_bt_arc75.mp4` (237 frames, 16 fps). Chunks 3–7 (frames 45–124) use
the "frozen" scene prompt from `config.json`. Other shots: `"window": ["45-124:orbit360"]` (full orbit)
or `["45-124:ring:r_ratio=0.3"]` (circular camera path).
