"""Camera trajectory generator.

Convention (OpenCV): c2w [N,4,4], +x right, +y down, +z forward. The world frame is the
source camera of frame 0, so pose 0 is always identity. A case with T source frames uses
N = T - 8 poses (one per generated frame). The last DWELL poses repeat the final pose.

Motions are parameterised by a 3D pivot (the subject; estimated from depth with
`--pivot auto`) so that the same design transfers across scenes of different scale.
"""
import argparse
import json
import os

import numpy as np

from .config import K4, DWELL, case_dir

# ----------------------------------------------------------------------------- basics


def ease5(t):
    t = np.clip(np.asarray(t, np.float64), 0, 1)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def ease_cos(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.asarray(t, np.float64))


def yaw_rot(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def pitch_rot(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def rot_yp(yaw, pitch):
    return yaw_rot(yaw) @ pitch_rot(pitch)


def look_at(pos, focus):
    z = focus - pos
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def rel_look(pos, pivot):
    """Orientation that looks at `pivot` and equals identity at the origin."""
    return look_at(pos, pivot) @ look_at(np.zeros(3), pivot).T


def split(n, weights):
    w = np.asarray(weights, np.float64)
    raw = n * w / w.sum()
    ns = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - ns))[: n - ns.sum()]:
        ns[i] += 1
    return ns.tolist()


def seg(n_list, fn_list):
    return np.concatenate([fn(ease5(np.linspace(0, 1, n))) for n, fn in zip(n_list, fn_list)])


def orbit_pose(yaw_deg, pivot):
    """Rigid rotation about the vertical axis through the pivot (subject stays centred)."""
    return pivot + yaw_rot(yaw_deg) @ (-pivot), yaw_rot(yaw_deg)

# ----------------------------------------------------------------------------- motions
# Every motion returns (pos [n,3], R [n,3,3]) and starts at identity.


def m_arc(n, pivot, amp=75.0, weights=(1, 2, 1)):
    """0 -> -amp -> +amp -> 0 around the pivot (constant distance, looking at the subject)."""
    n1, n2, n3 = split(n, list(weights))
    yaws = seg([n1, n2, n3], [lambda u: -amp * u, lambda u: -amp + 2 * amp * u, lambda u: amp * (1 - u)])
    P, R = zip(*[orbit_pose(y, pivot) for y in yaws])
    return np.stack(P), np.stack(R)


def m_arc_rep(n, pivot, amp=45.0, reps=2, side=0):
    P, R = [], []
    for k in split(n, [1.0] * reps):
        if side == 0:
            n1, n2, n3 = split(k, [1, 2, 1])
            yaws = seg([n1, n2, n3], [lambda u: -amp * u, lambda u: -amp + 2 * amp * u, lambda u: amp * (1 - u)])
        else:
            n1, n2 = split(k, [1, 1])
            yaws = seg([n1, n2], [lambda u: side * amp * u, lambda u: side * amp * (1 - u)])
        p, r = zip(*[orbit_pose(y, pivot) for y in yaws])
        P.append(np.stack(p))
        R.append(np.stack(r))
    return np.vstack(P), np.vstack(R)


def m_ring(n, pivot, r_ratio=0.3, ry=1.0):
    """Camera moves on a circle in the image plane (up -> right -> down -> left) and keeps
    looking at the pivot. r_ratio is the radius relative to the pivot distance."""
    r = r_ratio * np.linalg.norm(pivot)
    center = np.array([0.0, -ry * r, 0.0])
    phi = 2 * np.pi * ease5(np.linspace(0, 1, n))
    pos = np.stack([center + r * np.array([np.sin(p), ry * np.cos(p), 0.0]) for p in phi])
    pos[0] = 0.0
    pos[-1] = 0.0
    return pos, np.stack([rel_look(p, pivot) for p in pos])


def m_combo(n, pivot, dolly=0.5, yaw=55.0, weights=(3, 2, 4, 5, 4)):
    """Dolly in along the view ray and back, then arc -yaw -> +yaw -> 0."""
    n_in, n_out, n1, n2, n3 = split(n, list(weights))
    k = seg([n_in, n_out], [lambda u: dolly * u, lambda u: dolly * (1 - u)])
    pos_d = np.outer(k, pivot)
    yaws = seg([n1, n2, n3], [lambda u: -yaw * u, lambda u: -yaw + 2 * yaw * u, lambda u: yaw * (1 - u)])
    p, r = zip(*[orbit_pose(y, pivot) for y in yaws])
    return np.concatenate([pos_d, np.stack(p)]), np.concatenate([np.tile(np.eye(3), (len(pos_d), 1, 1)), np.stack(r)])


def m_dash(n, pivot, ratio=0.6, hold=12, tail=12):
    """Fast dolly along the view ray (ratio < 0 = pull back), hold, return, rest."""
    n_go = (n - hold - tail) // 2
    n_back = n - hold - tail - n_go
    k = np.concatenate([ratio * ease5(np.linspace(0, 1, n_go)), np.full(hold, ratio),
                        ratio * (1 - ease5(np.linspace(0, 1, n_back))), np.zeros(tail)])
    return np.outer(k, pivot), np.tile(np.eye(3), (n, 1, 1))


def m_fwdback(n, pivot, in_ratio=0.45, out_ratio=0.30):
    n1, n2, n3, n4 = split(n, [1.1, 1.0, 1.0, 0.9])
    k = np.concatenate([in_ratio * ease5(np.linspace(0, 1, n1)), in_ratio * (1 - ease5(np.linspace(0, 1, n2))),
                        -out_ratio * ease5(np.linspace(0, 1, n3)), -out_ratio * (1 - ease5(np.linspace(0, 1, n4)))])
    return np.outer(k, pivot), np.tile(np.eye(3), (n, 1, 1))


def m_truck(n, pivot, amp=0.30):
    """Sideways translation right -> back -> left -> back, then a small rise and return."""
    d = np.linalg.norm(pivot)
    n1, n2, n3, n4, n5 = split(n, [1, 1, 1, 1, 0.8])
    xs = np.concatenate([amp * d * np.sin(np.pi / 2 * ease5(np.linspace(0, 1, n1))),
                         amp * d * np.cos(np.pi / 2 * ease5(np.linspace(0, 1, n2))),
                         -amp * d * np.sin(np.pi / 2 * ease5(np.linspace(0, 1, n3))),
                         -amp * d * np.cos(np.pi / 2 * ease5(np.linspace(0, 1, n4))), np.zeros(n5)])
    ys = np.concatenate([np.zeros(n1 + n2 + n3 + n4), -0.12 * d * np.sin(np.pi * ease5(np.linspace(0, 1, n5)))])
    pos = np.stack([xs, ys, np.zeros(n)], 1)
    pos[0] = 0
    return pos, np.tile(np.eye(3), (n, 1, 1))


def m_rot(n, pivot, amp=45.0):
    """Pure yaw sweep in place: 0 -> -amp -> 0 -> +amp -> 0."""
    n4 = split(n, [1, 1, 1, 1])
    yaws = seg(n4, [lambda u: -amp * u, lambda u: -amp * (1 - u), lambda u: amp * u, lambda u: amp * (1 - u)])
    return np.zeros((n, 3)), np.stack([yaw_rot(y) for y in yaws])


def m_canonical(n, pivot, action="pan_left", magnitude=30.0):
    """One-way eased motion (cosine easing): pan/tilt in degrees, dolly/truck in units of
    the pivot distance, orbit_cw/orbit_ccw in degrees around the pivot."""
    D = float(np.linalg.norm(pivot))
    pos, R = np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1))
    for i, u in enumerate(ease_cos(np.arange(n) / (n - 1))):
        if action in ("pan_left", "pan_right"):
            R[i] = yaw_rot(-magnitude * u if action == "pan_left" else magnitude * u)
        elif action in ("tilt_up", "tilt_down"):
            R[i] = pitch_rot(magnitude * u if action == "tilt_up" else -magnitude * u)
        elif action in ("dolly_in", "dolly_out"):
            pos[i, 2] = (magnitude if action == "dolly_in" else -magnitude) * u * D
        elif action in ("truck_left", "truck_right"):
            pos[i, 0] = (-magnitude if action == "truck_left" else magnitude) * u * D
        elif action in ("orbit_cw", "orbit_ccw"):
            a = magnitude * u * (1.0 if action == "orbit_cw" else -1.0)
            pos[i], R[i] = orbit_pose(a, np.array([0.0, 0.0, D]))
        else:
            raise ValueError(action)
    return pos, R


MOTIONS = {
    "arc": m_arc, "arc_rep": m_arc_rep, "ring": m_ring, "combo": m_combo,
    "dash": m_dash, "fwdback": m_fwdback, "truck": m_truck, "rot": m_rot, "canonical": m_canonical,
}

# ----------------------------------------------------------------------------- schedules


def pan_tilt_schedule(plan, n):
    """plan: list of ("hold", k) / ("pan" | "tilt", deg, k). Each move is cosine-eased from
    the current angle to the target (pan > 0 turns right, tilt > 0 looks up); the other axis
    keeps its angle. Returns per-frame (yaw, pitch) in degrees."""
    ang = {"pan": [], "tilt": []}
    cur = {"pan": 0.0, "tilt": 0.0}
    for item in plan:
        if item[0] == "hold":
            k = item[1] if item[1] > 0 else max(0, n - len(ang["pan"]))
            for ax in ang:
                ang[ax] += [cur[ax]] * k
            continue
        ax, tgt, k = item
        assert ax in ang, f"unknown plan item {ax!r}; use hold / pan / tilt"
        u = ease_cos(np.linspace(0, 1, k + 1))[1:]
        ang[ax] += list(cur[ax] + (tgt - cur[ax]) * u)
        other = "tilt" if ax == "pan" else "pan"
        ang[other] += [cur[other]] * k
        cur[ax] = float(tgt)
    assert len(ang["pan"]) == n, f"schedule covers {len(ang['pan'])} frames, expected {n}"
    return np.asarray(ang["pan"], np.float64), np.asarray(ang["tilt"], np.float64)


def explore(segs, n, pivot):
    """Keyframe schedule: list of (k_frames, {theta, x, y, z, yaw, pitch}). theta orbits the
    pivot, x/y/z translate in units of the pivot distance, yaw/pitch turn the head."""
    tot = sum(k for k, _ in segs)
    if tot != n:
        raw = [k * n / tot for k, _ in segs]
        ks = [max(1, int(r)) for r in raw]
        for i in sorted(range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=True)[:n - sum(ks)]:
            ks[i] += 1
        segs = [(k, t) for k, (_, t) in zip(ks, segs)]
    keys = ("theta", "x", "y", "z", "yaw", "pitch")
    cur = dict.fromkeys(keys, 0.0)
    d = float(np.linalg.norm(pivot))
    pos, R = [], []
    for k, tgt in segs:
        nxt = dict(cur, **tgt)
        for a in ease5(np.linspace(0, 1, k + 1))[1:]:
            p = {kk: cur[kk] + (nxt[kk] - cur[kk]) * a for kk in keys}
            base = pivot + yaw_rot(p["theta"]) @ (-pivot)
            pos.append(base + np.array([p["x"], p["y"], p["z"]]) * d)
            R.append(rot_yp(p["theta"] + p["yaw"], p["pitch"]))
        cur = nxt
    pos = np.concatenate([np.zeros((1, 3)), np.asarray(pos)[:-1]])
    R = np.concatenate([np.eye(3)[None], np.asarray(R)[:-1]])
    return pos, R

# ----------------------------------------------------------------------------- assembly


def pad_dwell(pos, R, n_traj):
    n = len(pos)
    c2w = np.tile(np.eye(4), (n_traj, 1, 1))
    c2w[:n, :3, :3] = R
    c2w[:n, :3, 3] = pos
    c2w[n:] = c2w[n - 1]
    assert np.allclose(c2w[0], np.eye(4)), "first pose must be identity"
    return c2w


def build_windows(n_traj, windows, pivot):
    """Bullet time: identity everywhere except inside [lo, hi] windows, each running one
    motion that starts and ends at identity."""
    pos = np.zeros((n_traj, 3))
    R = np.tile(np.eye(3), (n_traj, 1, 1))
    for lo, hi, name, kw in windows:
        assert 0 < lo <= hi < n_traj, (lo, hi, n_traj)
        p, r = MOTIONS[name](hi - lo + 1, pivot, **kw)
        pos[lo:hi + 1], R[lo:hi + 1] = p, r
    c2w = np.tile(np.eye(4), (n_traj, 1, 1))
    c2w[:, :3, :3] = R
    c2w[:, :3, 3] = pos
    assert np.allclose(c2w[0], np.eye(4))
    return c2w


def subject_pivot(case, frame, depth_npz="depth.npz"):
    """3D pivot of the subject on `frame`: the nearest 12% of pixels inside the central
    80% of the image (excluding the bottom rows) back-projected at their median depth."""
    import torch
    from .depth_warp import DepthWarper
    d = np.load(os.path.join(case, depth_npz))["disp"].astype(np.float32)
    T = d.shape[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dw = DepthWarper(d, K4_frame(), np.tile(np.eye(4), (T, 1, 1)), device=dev, depth_smooth=True)
    dep = dw.depth[frame].float().cpu().numpy()
    H, W = dep.shape
    ys, xs = np.mgrid[0:H, 0:W]
    roi = (ys >= 0.08 * H) & (ys < 0.88 * H) & (xs >= 0.10 * W) & (xs < 0.90 * W)
    thr = np.quantile(dep[roi], 0.12)
    m = roi & (dep <= thr)
    u, v = float(xs[m].mean()), float(ys[m].mean())
    z = float(np.median(dep[m]))
    fx, fy, cx, cy = K4_frame()
    return np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])


def K4_frame():
    """Intrinsics at the 832x464 working resolution (intrinsics.npy is stored at 832x480)."""
    fx, fy, cx, cy = K4
    return fx, fy * 464 / 480, cx, cy * 464 / 480


def save_traj(tdir, c2w, info):
    os.makedirs(tdir, exist_ok=True)
    np.save(os.path.join(tdir, "poses.npy"), c2w.astype(np.float32))
    np.save(os.path.join(tdir, "intrinsics.npy"), np.tile(K4, (len(c2w), 1)).astype(np.float32))
    Rs = c2w[:, :3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(Rs, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    dpos = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
    info = dict(info, n=int(len(c2w)), max_rot_deg=float(ang.max()), max_step=float(dpos.max()),
                path_len=float(dpos.sum()))
    json.dump(info, open(os.path.join(tdir, "traj.json"), "w"), indent=1, default=str)
    print(f"{tdir}: {len(c2w)} poses, max rotation {ang.max():.0f} deg, max step {dpos.max():.2f}")
    return info


def parse_motion(spec):
    """'arc:amp=30,weights=1-2-1' -> ('arc', {...}); numbers parsed, '-' separated -> tuple."""
    name, _, args = spec.partition(":")
    assert name in MOTIONS, f"unknown motion {name!r}; choose from {', '.join(MOTIONS)}"
    kw = {}
    for a in [x for x in args.split(",") if x]:
        k, _, v = a.partition("=")
        if "-" in v and not v.lstrip("-").replace(".", "").isdigit():
            kw[k] = tuple(float(t) for t in v.split("-"))
        else:
            try:
                kw[k] = int(v) if v.lstrip("-").isdigit() else float(v)
            except ValueError:
                kw[k] = v
    return name, kw


def resolve_pivot(case, spec):
    if spec.startswith("auto"):
        frame = int(spec.split(":")[1]) if ":" in spec else 0
        pv = subject_pivot(case, frame)
        print(f"pivot (auto, frame {frame}): {pv.round(3)}")
        return pv
    v = [float(x) for x in spec.split(",")]
    return np.array(v if len(v) == 3 else [0.0, 0.0, v[0]])


def main():
    ap = argparse.ArgumentParser(description="Generate a camera trajectory for a case.")
    ap.add_argument("--case", required=True, help="case name or directory")
    ap.add_argument("--name", required=True, help="trajectory name -> <case>/traj_<name>/")
    ap.add_argument("--pivot", default="auto:0", help="'auto[:frame]' | 'x,y,z' | 'z'")
    ap.add_argument("--motion", default=None, help="whole-clip motion, e.g. 'arc:amp=30' 'rot:amp=45' "
                    "'ring:r_ratio=0.3' 'fwdback' 'truck' 'combo' 'canonical:action=pan_left,magnitude=30' "
                    f"(motions: {', '.join(MOTIONS)})")
    ap.add_argument("--window", action="append", default=[],
                    help="bullet time: 'lo-hi:motion[:args]', repeatable; identity outside")
    ap.add_argument("--pan_tilt", default=None,
                    help="in-place pan/tilt schedule 'hold:20,pan:-30:30,pan:30:60,pan:0:30,tilt:-20:30,...' "
                         "(item = hold:frames | pan:deg:frames | tilt:deg:frames)")
    ap.add_argument("--explore", default=None, help="JSON list of [frames, {theta,x,y,z,yaw,pitch}]")
    a = ap.parse_args()
    case = case_dir(a.case)
    T = np.load(os.path.join(case, "frames.npy"), mmap_mode="r").shape[0]
    n_traj = T - 8
    pivot = resolve_pivot(case, a.pivot)
    info = dict(pivot=[round(float(x), 4) for x in pivot])
    if a.window:
        wins = []
        for w in a.window:
            rng, _, mot = w.partition(":")
            lo, hi = (int(x) for x in rng.split("-"))
            name, kw = parse_motion(mot)
            wins.append((lo, hi, name, kw))
        c2w = build_windows(n_traj, wins, pivot)
        info.update(kind="bullet", windows=[[lo, hi, nm, kw] for lo, hi, nm, kw in wins])
    elif a.pan_tilt:
        plan = []
        for item in a.pan_tilt.split(","):
            p = item.split(":")
            plan.append(("hold", int(p[1])) if p[0] == "hold" else (p[0], float(p[1]), int(p[2])))
        yaw, pitch = pan_tilt_schedule(plan, n_traj - DWELL)
        R = np.stack([rot_yp(y, p) for y, p in zip(yaw, pitch)])
        c2w = pad_dwell(np.zeros((len(yaw), 3)), R, n_traj)
        info.update(kind="pan_tilt", plan=plan)
    elif a.explore:
        segs = [(int(k), dict(t)) for k, t in json.loads(a.explore)]
        pos, R = explore(segs, n_traj - DWELL, pivot)
        c2w = pad_dwell(pos, R, n_traj)
        info.update(kind="explore", segs=segs)
    else:
        assert a.motion, "give --motion, --window, --pan_tilt or --explore"
        name, kw = parse_motion(a.motion)
        pos, R = MOTIONS[name](n_traj - DWELL, pivot, **kw)
        pos = np.asarray(pos, np.float64)
        R = np.asarray(R, np.float64)
        pos[0], R[0] = 0, np.eye(3)
        c2w = pad_dwell(pos, R, n_traj)
        info.update(kind="motion", motion=name, args=kw)
    save_traj(os.path.join(case, f"traj_{a.name}"), c2w, info)


if __name__ == "__main__":
    main()
