# Robot bedroom — in-place rotation

`robot_bedroom.mp4` is a static shot of two robots making a bed (832x464, 16 fps, 245 frames). The
camera stays in place and turns: hold, turn left, hold, turn right, hold, left again, hold, back to
the start. Parts of the room the source never showed are painted by the model from the scene
prompt; the return to the start view shows how well the result stays consistent with the source.

```bash
python tools/run_example.py examples/robot_bedroom_rotation
```

`data/robot_bedroom/` holds the depth, prompt embeddings and the `rot70` trajectory. Output:
`workspace/robot_bedroom/out/robot_bedroom_rot70.mp4` (237 frames). The scene prompt is used for every
chunk here — the robots are visible from frame 0 and the rotation should not invent more of them.
Other shots: `"motion": "rot:amp=30"` (symmetric sweep) or `"motion": "arc:amp=30"` (arc around the bed).
