"""Geometric validity gates for warped source pixels (after UniWorld-View).

Three per-source-pixel tests decide whether a pixel may be splatted into the target view:
  round-trip visibility  - project to the target view and back; pixels that do not return to
                           the source grid were occluded / overwritten (triple re-projection);
  back-face culling      - surface normal must face the target camera (cos > 0);
  sliver triangles       - pixel-grid triangles with a minimum interior angle below 2.5 degrees
                           (depth discontinuities) are dropped.
All tests return a (H, W) float mask on the source grid that is multiplied into the splat mask.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .depth_warp import _bilinear_splat


def points_to_normals(points):
    p = points
    dx = torch.zeros_like(p)
    dy = torch.zeros_like(p)
    dx[:, :-1] = p[:, 1:] - p[:, :-1]
    dx[:, -1] = dx[:, -2]
    dy[:-1] = p[1:] - p[:-1]
    dy[-1] = dy[-2]
    n = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1, eps=1e-12)
    to_cam = F.normalize(-p, dim=-1, eps=1e-12)
    flip = (n * to_cam).sum(-1, keepdim=True) < 0
    return torch.where(flip, -n, n)


def sliver_mask(points, min_angle_deg=2.5):
    v00, v10 = points[:-1, :-1], points[1:, :-1]
    v01, v11 = points[:-1, 1:], points[1:, 1:]

    def min_ang(a, b, c):
        def ang(u, v):
            cu = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1) + 1e-12)
            return torch.arccos(cu.clamp(-1.0, 1.0))
        return torch.minimum(torch.minimum(ang(b - a, c - a), ang(c - b, a - b)), ang(a - c, b - c))

    thr = math.radians(float(min_angle_deg))
    b1 = min_ang(v00, v10, v01) < thr
    b2 = min_ang(v01, v10, v11) < thr
    out = torch.zeros(points.shape[:2], dtype=torch.bool, device=points.device)
    out[:-1, :-1] |= b1
    out[1:, :-1] |= b1 | b2
    out[:-1, 1:] |= b1 | b2
    out[1:, 1:] |= b2
    return out


class TriReproj:
    def __init__(self, normal_cos=0.0, sliver_deg=2.5):
        self.normal_cos = float(normal_cos)
        self.sliver_deg = float(sliver_deg)
        self.stats = []
        self._rays_t_cache = {}
        self._vc = {}      # per source view: point cloud, normals and sliver mask (target independent)

    def _view_cache(self, view_id, depth, base_keep):
        c = self._vc.get(view_id)
        if c is None or c["dep"] is not depth or c["bk"] is not base_keep:
            c = self._vc[view_id] = dict(dep=depth, bk=base_keep)
        return c

    def _rays_target(self, K, H, W, device):
        key = (round(float(K[0, 0]), 4), round(float(K[1, 1]), 4), round(float(K[0, 2]), 4),
               round(float(K[1, 2]), 4), H, W, str(device))
        r = self._rays_t_cache.get(key)
        if r is None:
            gy, gx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                    torch.arange(W, dtype=torch.float32), indexing="ij")
            pix = torch.stack([gx + 0.5, gy + 0.5, torch.ones_like(gx)], -1)
            r = pix.to(device) @ torch.linalg.inv(K).T.contiguous()
            self._rays_t_cache[key] = r
        return r

    @staticmethod
    def _proj(pts, K):
        proj = pts @ K.T
        z = proj[..., 2].clamp_min(1e-6)
        return proj[..., :2] / z.unsqueeze(-1) - 0.5, z

    def _roundtrip_mask(self, rays, depth, K, rel, base_keep):
        H, W = depth.shape
        dev = depth.device
        pts_s = rays * depth.unsqueeze(-1)
        pts_t = pts_s @ rel[:3, :3].T + rel[:3, 3]
        uv_t, z_t = self._proj(pts_t, K)
        m1 = (base_keep & (pts_t[..., 2] > 1e-3)).float()
        gy, gx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing="ij")
        u_n = (gx.to(dev) / max(W - 1, 1)) * 2.0 - 1.0
        v_n = (gy.to(dev) / max(H - 1, 1)) * 2.0 - 1.0
        inv_z = 1.0 / z_t.clamp_min(1e-3)
        src = torch.stack([u_n, v_n, inv_z], 0)
        fwd, m_fwd = _bilinear_splat(src.unsqueeze(0), m1[None, None], z_t.unsqueeze(0),
                                     uv_t.permute(2, 0, 1).unsqueeze(0), (H, W))
        fwd, m_fwd = fwd[0], m_fwd[0, 0]
        z_p = 1.0 / fwd[2].clamp_min(1e-6)
        pts_p = self._rays_target(K, H, W, dev) * z_p.unsqueeze(-1)
        rel_inv = torch.linalg.inv(rel)
        pts_back = pts_p @ rel_inv[:3, :3].T + rel_inv[:3, 3]
        uv_b, z_b = self._proj(pts_back, K)
        m3 = (m_fwd > 0) & (pts_back[..., 2] > 1e-3)
        _, m_back = _bilinear_splat(fwd[:2].unsqueeze(0), m3.float()[None, None], z_b.unsqueeze(0),
                                    uv_b.permute(2, 0, 1).unsqueeze(0), (H, W))
        m = m_back[0, 0]
        return torch.where(base_keep, m, torch.ones_like(m))

    def _front_mask(self, rays, depth, rel, base_keep, cache):
        if "pts" not in cache:
            cache["pts"] = rays * depth.unsqueeze(-1)
            cache["nrm"] = points_to_normals(cache["pts"])
        pts_s, n = cache["pts"], cache["nrm"]
        c_src = torch.linalg.inv(rel)[:3, 3]
        v = F.normalize(c_src.view(1, 1, 3) - pts_s, dim=-1, eps=1e-12)
        cos = (n * v).sum(-1)
        return torch.where(base_keep, (cos > self.normal_cos).float(), torch.ones_like(cos))

    def _break_keep(self, rays, depth, base_keep, cache):
        if "brk" not in cache:
            brk = sliver_mask(rays * depth.unsqueeze(-1), self.sliver_deg) & base_keep
            m = (~brk).float()
            cache["brk"] = torch.where(base_keep, m, torch.ones_like(m))
        return cache["brk"]

    @torch.no_grad()
    def gate(self, rays, depth, K, rel, base_keep, view_id=0, frame_id=0):
        vc = self._view_cache(view_id, depth, base_keep)
        parts = {}
        m_rt = self._roundtrip_mask(rays, depth, K, rel, base_keep)
        parts["rt"] = m_rt
        m = torch.ones_like(m_rt) * m_rt
        m_fr = self._front_mask(rays, depth, rel, base_keep, vc)
        parts["front"] = m_fr
        m = m * m_fr
        m_bk = self._break_keep(rays, depth, base_keep, vc)
        parts["break"] = m_bk
        m = m * m_bk
        base_n = float(base_keep.float().sum())
        st = dict(view=int(view_id), frame=int(frame_id),
                  keep=round(float((base_keep.float() * m).sum()) / max(base_n, 1.0), 4))
        for kk, vv in parts.items():
            st[f"drop_{kk}"] = round(1.0 - float((base_keep.float() * vv).sum()) / max(base_n, 1.0), 4)
        self.stats.append(st)
        self.last = dict(parts, gate=m)
        return m

    def summary(self):
        if not self.stats:
            return {}
        out = {"n": len(self.stats)}
        for kk in ("rt", "front", "break"):
            vals = [s[f"drop_{kk}"] for s in self.stats]
            out[f"drop_{kk}_mean"] = round(float(np.mean(vals)), 4)
        out["keep_min"] = round(float(np.min([s["keep"] for s in self.stats])), 4)
        return out
