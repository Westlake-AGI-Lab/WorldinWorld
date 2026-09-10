"""Conditioning tensors for the causal video model: first-frame (i2v) condition and Pluecker
camera embeddings for the target trajectory."""
import logging
import os

import numpy as np
import torch
from einops import rearrange
from wan.utils.cam_utils import (compute_relative_poses, get_Ks_transformed, get_plucker_embeddings,
                                 interpolate_camera_poses)


def build_y(pipe, frames, n_lat, h, w, lat_h, lat_w):
    """y = [mask (4ch) | VAE latent of frame 0 padded with zeros] as in the official i2v pipeline."""
    F = (n_lat - 1) * 4 + 1
    device = pipe.device
    img = torch.from_numpy(frames[0]).float().permute(2, 0, 1) / 127.5 - 1.0
    msk = torch.ones(1, F, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    y = pipe.vae.encode([torch.concat([img[None].transpose(0, 1), torch.zeros(3, F - 1, h, w)], dim=1).to(device)])[0]
    return torch.concat([msk, y])


def load_K4(traj_dir, h, w):
    Ks = torch.from_numpy(np.load(os.path.join(traj_dir, "intrinsics.npy"))).float()
    return get_Ks_transformed(Ks, height_org=480, width_org=832, height_resize=h, width_resize=w,
                              height_final=h, width_final=w)[0]


def build_plucker(traj_dir, n_lat, h, w, lat_h, lat_w, device, param_dtype, srccam_npz=None):
    """Trajectory -> (plucker embedding [1,C,n_lat,lat_h,lat_w], absolute latent poses [n_lat,4,4]).

    Poses are interpolated from per-frame to per-latent and converted to frame-wise relative
    motion (translation normalised by the largest step, as in the base model). With srccam_npz
    (per-frame c2w of the moving source camera) the embedding encodes src @ design so the model
    sees the true world path while the warp material keeps the source-camera world frame."""
    c2ws = np.load(os.path.join(traj_dir, "poses.npy"))
    n = len(c2ws)
    assert (n - 1) % (n_lat - 1) == 0, \
        f"{n} poses in {traj_dir} do not match the clip: expected {4 * (n_lat - 1) + 1} (= frames - 8)"
    Ks = load_K4(traj_dir, h, w)
    abs_poses = interpolate_camera_poses(src_indices=np.linspace(0, n - 1, n), src_rot_mat=c2ws[:, :3, :3],
                                         src_trans_vec=c2ws[:, :3, 3], tgt_indices=np.linspace(0, n - 1, n_lat))
    abs_pl = abs_poses
    if srccam_npz:
        src = np.load(srccam_npz)["c2w"].astype(np.float64)
        assert len(src) == n, f"source camera has {len(src)} poses, trajectory {n}"
        c2ws_pl = np.einsum("nij,njk->nik", src, c2ws.astype(np.float64))
        abs_pl = interpolate_camera_poses(src_indices=np.linspace(0, n - 1, n), src_rot_mat=c2ws_pl[:, :3, :3],
                                          src_trans_vec=c2ws_pl[:, :3, 3], tgt_indices=np.linspace(0, n - 1, n_lat))
        logging.info(f"[plucker] source camera folded into the embedding: {os.path.basename(srccam_npz)}")
    rel = compute_relative_poses(abs_pl.clone(), framewise=True)
    Ks = Ks.repeat(n_lat, 1)
    emb = get_plucker_embeddings(rel.to(device), Ks.to(device), h, w)
    emb = rearrange(emb, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=h // lat_h, c2=w // lat_w)[None]
    emb = rearrange(emb, 'b (f h w) c -> b c f h w', f=n_lat, h=lat_h, w=lat_w).to(param_dtype)
    return emb, abs_poses
