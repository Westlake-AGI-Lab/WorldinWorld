"""Depth-based forward warping of source frames into the target camera.

Each source frame is lifted with its (relative) depth and splatted into the target view with a
depth-weighted soft z-buffer (bilinear splatting, no mesh renderer). Pixels on depth edges,
strongly distorted pixels, back-facing / occluded pixels (tri_reproj) and pixels behind a
mesh skirt (skirt_occ) are removed; the remaining holes are left empty and reported through a
per-pixel visibility map that later becomes a per-token weight.

Coordinates: the world frame is the source camera of every frame (identity); poses.npy holds
target c2w matrices (OpenCV, +z forward). Depth is metric-free: the 5th percentile of inverse
disparity is anchored at ANCHOR_DEPTH so that trajectory translations are in a fixed unit.
"""
import numpy as np
import torch
import torch.nn.functional as F

ANCHOR_DEPTH = 10.0
ANCHOR_Q = 0.05
DEPTH_EPS = 0.1


class DepthWarper:
    def __init__(self, disp, K4, c2ws, device="cuda", depth_smooth=False, tri=None,
                 quad_reject=False, skirt=False, skirt_deg=2.5, src_hole_seq=None,
                 edge_thresh=0.1, edge_dilation=2, jump_thresh=0.08, jump_radius=2, soft_px=8,
                 supersample=2, aniso_max=2.0, area_band=(0.3, 6.0)):
        """disp (T,H,W): normalised inverse depth (1 = near); K4: (fx, fy, cx, cy) at frame
        resolution; c2ws (T,4,4): target poses. src_hole_seq (T,H,W) bool: source pixels to
        exclude per frame (video editing)."""
        with torch.amp.autocast("cuda", enabled=False):
            self.tri = tri
            self.skirt = bool(skirt)
            self.skirt_deg = float(skirt_deg)
            self.last_skirt = None
            self.quad_reject = bool(quad_reject)
            self.device = device
            self.fx, self.fy, self.cx, self.cy = [float(v) for v in K4]
            self.T, self.H, self.W = disp.shape
            self.soft_px = int(soft_px)
            self.ss = int(supersample)
            self.aniso_max = float(aniso_max)
            self.area_band = tuple(area_band)
            d = torch.from_numpy(np.ascontiguousarray(disp)).float()
            if depth_smooth:
                d = self._smooth_disp(d.to(device)).cpu()
            raw = 1.0 / (d + DEPTH_EPS)
            self.depth_scale = ANCHOR_DEPTH / float(torch.quantile(raw[::8].flatten(), ANCHOR_Q))
            self.depth = (raw * self.depth_scale).to(device)
            self.c2ws = torch.from_numpy(np.ascontiguousarray(c2ws)).float().to(device)
            self.w2cs = torch.linalg.inv(self.c2ws)
            self.keep = self._edge_keep_masks(d.to(device), edge_thresh, edge_dilation,
                                              jump_thresh, jump_radius)
            self._src_hole_seq = None
            if src_hole_seq is not None:
                mk = torch.as_tensor(np.ascontiguousarray(src_hole_seq)).bool().to(device)
                assert mk.ndim == 3 and tuple(mk.shape[1:]) == (self.H, self.W), mk.shape
                t_src = min(mk.shape[0], self.keep.shape[0])
                self.keep = self.keep.clone()
                for t in range(t_src):
                    self.keep[t] = self.keep[t] & (~mk[t])
                self._src_hole_seq = mk[:t_src]
            K = torch.eye(3)
            K[0, 0], K[1, 1] = self.fx, self.fy
            K[0, 2], K[1, 2] = self.cx, self.cy
            self.K = K.to(device)
            self.K_inv = torch.linalg.inv(K).to(device)
            gy, gx = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")
            self.grid = torch.stack([gx, gy], 0).float().to(device)
            self.rays = self._make_rays(self.H, self.W)
            self.rays_ss = self.rays if self.ss == 1 else self._make_rays(self.H * self.ss, self.W * self.ss)
            self.frames = None

    @torch.no_grad()
    def _smooth_disp(self, d, tw=5, sp_sigma=9.0):
        """Temporal median on static pixels + edge-preserving spatial low-pass."""
        T = d.shape[0]
        pad = tw // 2
        med = torch.empty_like(d)
        for lo in range(0, T, 32):
            hi = min(lo + 32, T)
            a = max(0, lo - pad)
            b = min(T, hi + pad)
            win = d[a:b].unfold(0, min(tw, b - a), 1)
            m = win.median(dim=-1).values
            for t in range(lo, hi):
                med[t] = m[min(max(t - pad - a, 0), m.shape[0] - 1)]
        std_t = d.std(dim=0, keepdim=True)
        s = std_t.flatten()
        s = s[:: max(1, s.numel() // 10_000_000)]
        p_lo = torch.quantile(s, 0.25)
        p_hi = torch.quantile(s, 0.75)
        w_move = ((std_t - p_lo) / (p_hi - p_lo).clamp_min(1e-8)).clamp(0, 1).expand_as(d)
        d = w_move * d + (1 - w_move) * med
        k = int(sp_sigma * 3) | 1
        x = torch.arange(k, device=d.device, dtype=torch.float32) - k // 2
        g1 = torch.exp(-x ** 2 / (2 * sp_sigma ** 2))
        g1 /= g1.sum()
        lp = F.conv2d(d.unsqueeze(1), g1.view(1, 1, 1, k), padding=(0, k // 2))
        lp = F.conv2d(lp, g1.view(1, 1, k, 1), padding=(k // 2, 0)).squeeze(1)
        resid = (d - lp).abs()
        r = resid.flatten()
        r = r[:: max(1, r.numel() // 10_000_000)]
        edge_keep = float(torch.quantile(r, 0.75))
        return torch.where(resid > edge_keep, d, lp)

    def _make_rays(self, h, w):
        sy, sx = self.H / h, self.W / w
        gy, gx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                                torch.arange(w, dtype=torch.float32), indexing="ij")
        pix = torch.stack([(gx + 0.5) * sx, (gy + 0.5) * sy, torch.ones_like(gx)], -1)
        return pix.to(self.device) @ self.K_inv.T

    @torch.no_grad()
    def _edge_keep_masks(self, disp, thresh, dilation, jump_thresh, radius):
        """Source pixels that are not on a disparity edge (Sobel gradient + local jump)."""
        keeps = []
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=disp.device).view(1, 1, 3, 3)
        ky = kx.transpose(-1, -2)
        for lo in range(0, self.T, 64):
            d = disp[lo:lo + 64].unsqueeze(1)
            gm = (F.conv2d(d, kx, padding=1) ** 2 + F.conv2d(d, ky, padding=1) ** 2).sqrt()
            gmax = gm.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
            edge = (gm / gmax) > thresh
            if dilation > 0:
                k = 2 * dilation + 1
                edge = F.max_pool2d(edge.float(), k, 1, dilation) > 0
            if jump_thresh > 0 and radius > 0:
                k = 2 * radius + 1
                edge |= (F.max_pool2d(d, k, 1, radius) + F.max_pool2d(-d, k, 1, radius)) > jump_thresh
            keeps.append(~edge.squeeze(1))
        return torch.cat(keeps)

    def _project(self, fi, rays, depth):
        pts_world = rays * depth.unsqueeze(-1)
        w2c = self.w2cs[fi]
        pts_cam = pts_world @ w2c[:3, :3].T + w2c[:3, 3]
        proj = pts_cam @ self.K.T
        z = proj[..., 2].clamp_min(1e-6)
        uv = proj[..., :2] / z.unsqueeze(-1) - 0.5
        return uv, z, pts_cam[..., 2] > 1e-3

    def _src_hole(self, fi):
        shs = self._src_hole_seq
        if shs is None or int(fi) >= shs.shape[0]:
            return None
        return shs[int(fi)] if bool(shs[int(fi)].any()) else None

    @torch.no_grad()
    def warp_frame(self, fi, ret_fid=False):
        """Warp source frame fi into target pose fi. Returns (rgb (H,W,3) in [-1,1], vis (H,W))
        and, with ret_fid, (fidelity, static-edge, depth) maps of the warped image."""
        with torch.amp.autocast("cuda", enabled=False):
            self.last_skirt = None
            return self._warp_frame_fp32(fi, ret_fid)

    def _warp_frame_fp32(self, fi, ret_fid=False):
        H, W = self.H, self.W
        uv, _, _ = self._project(fi, self.rays, self.depth[fi])
        flow_max = float((uv.permute(2, 0, 1) - self.grid).abs().max())
        rgb = self._rgb(fi)
        hole = self._src_hole(fi)
        if flow_max < 0.5 and hole is None:
            out = (rgb.permute(1, 2, 0).clone(), torch.ones(H, W, device=self.device))
            if ret_fid:
                out = out + (torch.ones(H, W, device=self.device), (~self.keep[fi]).float(),
                             self.depth[fi].clone())
            return out
        keep_fi = self.keep[fi]
        sz = (H * self.ss, W * self.ss)
        depth_s = F.interpolate(self.depth[fi][None, None], sz, mode="bilinear", align_corners=False)[0, 0]
        keep_s = F.interpolate(keep_fi[None, None].float(), sz, mode="nearest")[0, 0] > 0.5
        rgb_s = F.interpolate(rgb[None], sz, mode="bilinear", align_corners=False)[0]
        edge_static = (~keep_s).float()
        uv_s, z_s, front = self._project(fi, self.rays_ss, depth_s)
        keep_d, m_area = self._distortion_keep(uv_s)
        mask1 = (keep_s & front & keep_d).float()
        tri_full = None
        if self.tri is not None:
            tri_full = self.tri.gate(self.rays, self.depth[fi], self.K, self.w2cs[fi], keep_fi,
                                     view_id=0, frame_id=int(fi))
            mask1 = mask1 * (tri_full if tuple(tri_full.shape) == mask1.shape
                             else F.interpolate(tri_full[None, None], tuple(mask1.shape), mode="nearest")[0, 0])
        if self.quad_reject and self.ss > 1:
            valid_full = keep_fi
            if tri_full is not None:
                valid_full = valid_full & (tri_full > 0.5)
            mask1 = mask1 * quad_keep_upsample(valid_full, self.ss).float()
        fid_s = (1.0 / m_area.clamp_min(1e-6)).clamp(max=1.0)
        src = torch.cat([rgb_s, fid_s[None], edge_static[None], (1.0 / z_s.clamp_min(1e-3))[None]], 0)
        warped, mask2 = _bilinear_splat(src.unsqueeze(0), mask1[None, None], z_s.unsqueeze(0),
                                        uv_s.permute(2, 0, 1).unsqueeze(0), (H, W))
        mask2 = mask2[0, 0]
        self.last_skirt = None
        if self.skirt:
            from .skirt_occ import skirt_zbuf, skirt_occlude
            zs = skirt_zbuf(self.rays, self.depth[fi], self.w2cs[fi], self.K, (H, W), sliver_deg=self.skirt_deg)
            occ = skirt_occlude(zs, 1.0 / warped[0, 5].clamp_min(1e-4), mask2)
            if bool(occ.any()):
                mask2 = mask2 * (~occ).float()
                warped[0][:, occ] = 0.0
            self.last_skirt = occ
        k = 2 * self.soft_px + 1
        vis = mask2 * F.avg_pool2d(mask2[None, None], k, 1, self.soft_px)[0, 0]
        out = (warped[0, :3].permute(1, 2, 0), vis)
        if ret_fid:
            z_out = torch.where(mask2 > 0, 1.0 / warped[0, 5].clamp_min(1e-4), torch.zeros_like(mask2))
            out = out + (warped[0, 3].clamp(0, 1), warped[0, 4].clamp(0, 1), z_out)
        return out

    def _distortion_keep(self, uv):
        """Reject pixels whose local mapping is too anisotropic or too magnified/shrunk."""
        step = 1.0 / self.ss if uv.shape[0] != self.H else 1.0
        du = torch.gradient(uv[..., 0], dim=1)[0] / step
        dv = torch.gradient(uv[..., 1], dim=0)[0] / step
        a = self.aniso_max
        aniso = du.abs().clamp_min(1e-6) / dv.abs().clamp_min(1e-6)
        keep = (aniso > 1 / a) & (aniso < a)
        m = (du * dv).abs()
        keep &= (m > self.area_band[0]) & (m < self.area_band[1])
        return keep, m

    def _rgb(self, fi):
        f = torch.from_numpy(np.ascontiguousarray(self.frames[fi])).to(self.device)
        return f.permute(2, 0, 1).float() / 127.5 - 1.0

    def set_frames(self, frames):
        self.frames = frames


@torch.no_grad()
def quad_index_ss(H, W, ss, device):
    j = torch.arange(W * ss, dtype=torch.float32, device=device)
    i = torch.arange(H * ss, dtype=torch.float32, device=device)
    qx = torch.floor((j + 0.5) / ss - 0.5).clamp(0, W - 2).long()
    qy = torch.floor((i + 0.5) / ss - 0.5).clamp(0, H - 2).long()
    return qy[:, None] * (W - 1) + qx[None, :]


def quad_keep_upsample(keep, ss, _cache={}):
    """Super-sampled validity: a sample is valid only if all four corners of its quad are."""
    H, W = keep.shape
    q = keep[:-1, :-1] & keep[1:, :-1] & keep[:-1, 1:] & keep[1:, 1:]
    key = (H, W, ss, str(keep.device))
    idx = _cache.get(key)
    if idx is None:
        idx = _cache[key] = quad_index_ss(H, W, ss, keep.device)
    return q.reshape(-1)[idx]


def _bilinear_splat(frame1, mask1, depth1, trans_pos, out_hw):
    """Bilinear forward splatting with a soft depth-weighted z-buffer.
    frame1 (B,C,h,w), mask1 (B,1,h,w), depth1 (B,h,w), trans_pos (B,2,h,w) target pixel coords.
    Returns (warped (B,C,H,W), mask (B,1,H,W))."""
    b, c = frame1.shape[:2]
    h, w = out_hw
    tp_off = trans_pos + 1
    tp_floor = torch.floor(tp_off).long()
    tp_ceil = torch.ceil(tp_off).long()
    tp_off = torch.stack([tp_off[:, 0].clamp(0, w + 1), tp_off[:, 1].clamp(0, h + 1)], 1)
    tp_floor = torch.stack([tp_floor[:, 0].clamp(0, w + 1), tp_floor[:, 1].clamp(0, h + 1)], 1)
    tp_ceil = torch.stack([tp_ceil[:, 0].clamp(0, w + 1), tp_ceil[:, 1].clamp(0, h + 1)], 1)
    pw_nw = (1 - (tp_off[:, 1:2] - tp_floor[:, 1:2])) * (1 - (tp_off[:, 0:1] - tp_floor[:, 0:1]))
    pw_sw = (1 - (tp_ceil[:, 1:2] - tp_off[:, 1:2])) * (1 - (tp_off[:, 0:1] - tp_floor[:, 0:1]))
    pw_ne = (1 - (tp_off[:, 1:2] - tp_floor[:, 1:2])) * (1 - (tp_ceil[:, 0:1] - tp_off[:, 0:1]))
    pw_se = (1 - (tp_ceil[:, 1:2] - tp_off[:, 1:2])) * (1 - (tp_ceil[:, 0:1] - tp_off[:, 0:1]))
    log_d = torch.log1p(depth1.clamp(0, 1000))
    dw = torch.exp(log_d / log_d.max() * 50).unsqueeze(1)
    out = torch.zeros(b, h + 2, w + 2, c, device=frame1.device)
    wsum = torch.zeros(b, h + 2, w + 2, 1, device=frame1.device)
    f_cl = frame1.permute(0, 2, 3, 1)
    sel = (mask1.reshape(-1) > 0).nonzero(as_tuple=True)[0]
    bi_f = torch.arange(b, device=frame1.device)[:, None, None].expand(b, *mask1.shape[-2:]).reshape(-1)[sel]
    f_sel = f_cl.reshape(-1, c)[sel]
    for pw, ty, tx in ((pw_nw, tp_floor[:, 1], tp_floor[:, 0]),
                       (pw_sw, tp_ceil[:, 1], tp_floor[:, 0]),
                       (pw_ne, tp_floor[:, 1], tp_ceil[:, 0]),
                       (pw_se, tp_ceil[:, 1], tp_ceil[:, 0])):
        wgt = (pw * mask1 / dw).reshape(-1, 1)[sel]
        idx = (bi_f, ty.reshape(-1)[sel], tx.reshape(-1)[sel])
        out.index_put_(idx, f_sel * wgt, accumulate=True)
        wsum.index_put_(idx, wgt, accumulate=True)
    out = out.permute(0, 3, 1, 2)[:, :, 1:-1, 1:-1]
    wsum = wsum.permute(0, 3, 1, 2)[:, :, 1:-1, 1:-1]
    hole = wsum <= 0
    warped = torch.where(hole, torch.zeros_like(out), out / wsum).clamp(-1, 1)
    return warped, (~hole).float()


def fill_holes(rgb_np_u8, vis_np):
    """Fill holes with the colour of the nearest visible pixel, then smooth inside the hole."""
    import cv2
    hole0 = (vis_np <= 0).astype(np.uint8)
    if not hole0.any():
        return rgb_np_u8
    _, lab = cv2.distanceTransformWithLabels(hole0, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    valid = vis_np > 0
    lut = np.zeros((int(lab.max()) + 1, 3), np.uint8)
    lut[lab[valid]] = rgb_np_u8[valid]
    hb = hole0[..., None] > 0
    out = np.where(hb, lut[lab], rgb_np_u8)
    sm = out
    for _ in range(2):
        sm = cv2.blur(sm, (5, 5))
        sm = np.where(hb, sm, out)
    return np.where(hb, sm, rgb_np_u8).astype(np.uint8)


def token_weights(vis4, gh, gw, quant=8, agg="mean"):
    """Per-token weight from the pixel visibility maps of the 4 frames that make one latent."""
    v = torch.stack(vis4)
    v = v.mean(0) if agg == "mean" else v.amin(0)
    H, W = v.shape
    ph, pw = H // gh, W // gw
    w = v[:gh * ph, :gw * pw].reshape(gh, ph, gw, pw).mean(dim=(1, 3))
    w = torch.round(w * quant) / quant
    return w.flatten().cpu().numpy().astype(np.float32)
