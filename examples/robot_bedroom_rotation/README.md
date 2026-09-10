# Robot bedroom — in-place pan and tilt

`robot_bedroom.mp4` is a static shot of two robots making a bed (832x464, 16 fps, 245 frames). The
camera stays in place and looks around without pausing: it tilts down and up (−26° / +18°), then
pans left and right (±38°) and returns to the start view. Parts of the room the source never
showed are painted by the model from the scene prompt; the return to the start view shows how
well the result stays consistent with the source.

```bash
python tools/run_example.py examples/robot_bedroom_rotation
```

`data/robot_bedroom/` holds the depth, prompt embeddings and the `pantilt` trajectory. Output:
`workspace/robot_bedroom/out/robot_bedroom_pantilt.mp4` (237 frames). The scene prompt is used for
every chunk here — the robots are visible from frame 0 and the rotation should not invent more of
them. Other shots: edit the `pan_tilt` plan (`hold:frames`, `pan:deg:frames`, `tilt:deg:frames`;
pan > 0 turns right, tilt > 0 looks up), or use `"motion": "rot:amp=30"` (symmetric pan sweep) or
`"motion": "arc:amp=30"` (arc around the bed).
