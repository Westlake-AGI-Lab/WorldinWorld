"""Chunk-wise generation with warped-source evidence (the World-in-World re-cinematography loop).

The base model generates 4 latents (16 frames) per chunk with a rolling KV cache of
6 sink + 8 window + 4 current latent slots. For every chunk:
  1. the source frames of that time span are warped into the target cameras, hole-filled and
     VAE-encoded; one extra DiT forward (with the chunk's RoPE positions) turns them into
     K/V evidence that is appended to the attention context on the first denoising steps;
  2. CGAR links generated tokens to the clean source tokens of the same moment;
  3. after the 4 denoising steps a final forward with the denoised latents writes the chunk
     into the KV cache (self-generated history, official causal-fast semantics).
"""
import logging
import os

import numpy as np
import torch

from .config import CHUNK, SINK_LAT, WINDOW_LAT
from .corr_bias import SourceQueryPlanner
from .depth_warp import DepthWarper, fill_holes, token_weights
from .plucker import build_plucker, build_y, load_K4
from .tri_reproj import TriReproj

MIN_TOKENS = 16
TIMESTEP_IDX = (0, 250, 500, 750)


class Runner:
    def __init__(self, pipe, frames, source_latents, traj_dir, contexts, seed=42, shift=4.0, srccam_npz=None):
        self.pipe = pipe
        self.traj_dir = traj_dir
        self.device = pipe.device
        self.frames = frames
        self.src_lat = source_latents.to(self.device)
        self.n_lat = source_latents.shape[1]
        self.lat_h, self.lat_w = source_latents.shape[2], source_latents.shape[3]
        self.h, self.w = self.lat_h * 8, self.lat_w * 8
        self.fs = self.lat_h * self.lat_w // 4
        self.kv_size = self.fs * WINDOW_LAT
        self.contexts = [c.to(self.device) for c in contexts]
        self.seed = seed
        self.n_gen = (self.n_lat // CHUNK) * CHUNK
        if self.n_gen + 2 > self.n_lat:
            self.n_gen -= CHUNK
        self.n_chunks = self.n_gen // CHUNK
        self.y = build_y(pipe, frames, self.n_lat, self.h, self.w, self.lat_h, self.lat_w)
        self.plucker, self.abs_poses = build_plucker(traj_dir, self.n_gen, self.h, self.w, self.lat_h, self.lat_w,
                                                    self.device, pipe.param_dtype, srccam_npz=srccam_npz)
        self.pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=shift)
        self.timesteps = self.pipe.scheduler.timesteps[list(TIMESTEP_IDX)]
        logging.info(f"sampling: shift={shift} t={[round(float(x), 1) for x in self.timesteps]}")
        self.log = []
        self.evid = {"in": {}, "w": {}}
        self._cross_caches = {}
        self._sink_lat_now = SINK_LAT
        self.dw = None
        self.edit_cut = None      # (T,H,W) bool: source pixels excluded from the warp material
        self.edit_paste = None    # dict(rgb (T,H,W,3), mask (T,H,W) bool, w)

    # ------------------------------------------------------------------ model plumbing
    def _cross_cache(self, ctx_id):
        if ctx_id not in self._cross_caches:
            m = self.pipe.model.config
            self._cross_caches[ctx_id] = self.pipe._initialize_crossattn_cache(
                num_layers=m.num_layers, shape=[1, 512, m.num_heads, m.dim // m.num_heads],
                dtype=self.pipe.pipe_dtype, device=self.device)
            return self._cross_caches[ctx_id], True
        return self._cross_caches[ctx_id], False

    def _forward(self, x_chunk, t_val, lo, kv_cache, ctx_id=0):
        n = x_chunk.shape[1]
        timestep = torch.stack([t_val]).to(self.device) if torch.is_tensor(t_val) \
            else torch.tensor([float(t_val)], device=self.device)
        cross_cache, first = self._cross_cache(ctx_id)
        return self.pipe.model(
            x=[x_chunk.to(self.device)], t=timestep, context=[self.contexts[ctx_id]], seq_len=n * self.fs,
            y=[self.y[:, lo:lo + n]], dit_cond_dict={"c2ws_plucker_emb": (self.plucker[:, :, lo:lo + n],)},
            kv_cache=kv_cache, crossattn_cache=cross_cache, current_start=lo * self.fs,
            max_attention_size=self.kv_size, frame_seqlen=self.fs, cross_attn_first_call=first)[0]

    def _new_kv_cache(self, n_tokens=None):
        m = self.pipe.model.config
        return self.pipe._initialize_self_kv_cache(
            num_layers=m.num_layers, shape=[1, n_tokens or self.kv_size, m.num_heads, m.dim // m.num_heads],
            dtype=self.pipe.pipe_dtype, device=self.device)

    def _set_sink_lat(self, n):
        for blk in self.pipe.model.blocks:
            blk.self_attn.sink_size = n
        self._sink_lat_now = n

    @torch.no_grad()
    def _rebase_cache(self, kv_cache, c):
        """Sink/window bookkeeping before chunk c: chunk 0 has no history, chunk 1 keeps the
        4 latents of chunk 0 as sink, from chunk 4 on the oldest window latents are evicted."""
        fs = self.fs
        if c == 0:
            self._set_sink_lat(0)
            return
        if c == 1:
            self._set_sink_lat(4)
            return
        if c in (2, 3):
            self._set_sink_lat(SINK_LAT)
            return
        n_evict = 2 if c == 4 else 4
        lo_e, hi_e = SINK_LAT, SINK_LAT + n_evict
        for layer in kv_cache:
            le = int(layer["local_end_index"].item())
            kk = layer["k"][:, hi_e * fs:le].clone()
            vv = layer["v"][:, hi_e * fs:le].clone()
            layer["k"][:, lo_e * fs:lo_e * fs + kk.shape[1]] = kk
            layer["v"][:, lo_e * fs:lo_e * fs + vv.shape[1]] = vv
            layer["local_end_index"].fill_(le - n_evict * fs)
        self._set_sink_lat(SINK_LAT)

    def _block_kv(self, latents, lo, ctx_id):
        """Run the DiT once on `latents` [16,n,lh,lw] with the RoPE positions of chunk lo and
        return the per-layer (k, v) of that block (no evidence hooks active)."""
        n = latents.shape[1]
        tmp = self._new_kv_cache(n * self.fs)
        for layer in tmp:
            layer["global_end_index"].fill_(lo * self.fs)
        saved = [blk.self_attn.kv_bank for blk in self.pipe.model.blocks]
        for blk in self.pipe.model.blocks:
            blk.self_attn.kv_bank = None
        self._forward(latents, 0.0, lo, tmp, ctx_id=ctx_id)
        for blk, kb in zip(self.pipe.model.blocks, saved):
            blk.self_attn.kv_bank = kb
        return tmp

    # ------------------------------------------------------------------ warp material
    def setup_warp(self, disp, tracks_npz=None, skirt=True, srcq_gamma=2.0, vote_min=4, edit_hole_seq=None):
        K4 = load_K4(self.traj_dir, self.h, self.w)
        c2ws = np.load(os.path.join(self.traj_dir, "poses.npy"))
        self.dw = DepthWarper(disp, K4.numpy(), c2ws, device=self.device, depth_smooth=True, tri=TriReproj(),
                              quad_reject=True, skirt=skirt, src_hole_seq=edit_hole_seq)
        self.dw.set_frames(self.frames)
        self.log.append({"depth_warper": dict(scale=round(self.dw.depth_scale, 3), skirt=skirt)})
        self.planner = None
        if tracks_npz and srcq_gamma > 0:
            self.planner = SourceQueryPlanner(tracks_npz, device=self.device, gamma=srcq_gamma, vote_min=vote_min)
            logging.info(f"[cgar] source queries on: gamma={srcq_gamma} vote_min={vote_min}")

    def _warp_u8(self, fi):
        """Warped source frame fi as uint8 (holes filled with neighbouring colours) + visibility."""
        rgb, vis, _, _, _ = self.dw.warp_frame(fi, ret_fid=True)
        u8 = ((rgb.cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
        vis_np = (vis > 0).cpu().numpy().astype(np.float32)
        if self.edit_cut is not None:
            cm = self.edit_cut[min(fi, len(self.edit_cut) - 1)]
            vis_np[cm] = 0.0
            vis = vis.clone()
            vis[torch.from_numpy(cm).to(vis.device)] = 0.0
        ep = self.edit_paste
        if ep is not None:
            em = ep["mask"][min(fi, len(ep["mask"]) - 1)]
            u8 = u8.copy()
            u8[em] = ep["rgb"][min(fi, len(ep["rgb"]) - 1)][em]
            vis_np[em] = ep["w"]
            vis = vis.clone()
            vis[torch.from_numpy(em).to(vis.device)] = ep["w"]
        return fill_holes(u8, vis_np), vis

    def _prewarm(self):
        """Warp the whole clip once and VAE-encode it; chunks slice the resulting latents."""
        if hasattr(self, "_glat"):
            return
        n_fr = min(4 * (self.n_lat - 1) + 1, len(self.dw.c2ws), len(self.frames))
        fr_all = [self._warp_u8(fi)[0] for fi in range(n_fr)]
        vid = torch.from_numpy(np.stack(fr_all)).float().permute(3, 0, 1, 2) / 127.5 - 1.0
        self._glat = self.pipe.vae.encode([vid.to(self.device)])[0]
        self._gframes = fr_all
        logging.info(f"[material] warped {n_fr} frames -> {self._glat.shape[1]} latents")
        torch.cuda.empty_cache()

    def _material(self, lo):
        """Evidence for chunk lo: latents [16 x 4], selected token ids and log weights per latent."""
        if lo == 0:
            f0, n_ctx = 0, 13
            lat_fr = [[0]] + [[4 * m - 3 + j for j in range(4)] for m in range(1, CHUNK)]
        else:
            f0, n_ctx = 4 * lo - 4, 4 * CHUNK + 1
            lat_fr = [[1 + 4 * m + j for j in range(4)] for m in range(CHUNK)]
        frames17, vis17 = [], []
        for i in range(n_ctx):
            fi = min(f0 + i, len(self.frames) - 1)
            u8, vis = self._warp_u8(fi)
            frames17.append(u8)
            vis17.append(vis)
        logging.info(f"[warp] chunk{lo // CHUNK}: {self.dw.tri.summary()}")
        gh, gw = self.lat_h // 2, self.lat_w // 2
        tok_idx, tok_logw = [], []
        for m in range(CHUNK):
            w = token_weights([vis17[i] for i in lat_fr[m]], gh, gw, quant=8, agg="mean")
            nz = np.nonzero(w > 0)[0]
            if len(nz) < MIN_TOKENS:
                tok_idx.append(None)
                tok_logw.append(None)
            else:
                tok_idx.append(nz)
                tok_logw.append(np.log(w[nz]))
            wg = (w if tok_idx[m] is not None else np.zeros_like(w)).reshape(gh, gw)
            for rel in lat_fr[m]:
                fi_e = min(f0 + rel, len(self.frames) - 1)
                self.evid["in"][fi_e] = frames17[rel]
                self.evid["w"][fi_e] = wg
        self._prewarm()
        assert lo + CHUNK <= self._glat.shape[1]
        assert (frames17[0] == self._gframes[min(f0, len(self._gframes) - 1)]).all()
        wl = [self._glat[:, lo + m] for m in range(CHUNK)]
        return wl, tok_idx, tok_logw

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, ph_bank, ph_steps=(0, 1, 2), chunk_ctx_ids=None, max_chunks=None, srcq_cpu=True):
        assert self.dw is not None, "call setup_warp() first"
        if chunk_ctx_ids is None:
            chunk_ctx_ids = [0] * self.n_chunks
        assert len(chunk_ctx_ids) == self.n_chunks
        kv_cache = self._new_kv_cache()
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(self.seed)
        noise = torch.randn(16, self.n_gen, self.lat_h, self.lat_w, dtype=torch.float32, generator=seed_g,
                            device=self.device)
        n_run = self.n_chunks if max_chunks is None else min(self.n_chunks, int(max_chunks))
        pred_chunks = []
        dsg = ph_bank.ph_dsg
        with torch.amp.autocast("cuda", dtype=self.pipe.param_dtype):
            for c in range(n_run):
                lo = CHUNK * c
                self._rebase_cache(kv_cache, c)
                ph_bank.sink_tokens = self._sink_lat_now * self.fs
                cid = chunk_ctx_ids[c]
                ph_bank.stage_kv(None)
                ph_bank._corr_kv_s = ph_bank._corr_sel_s = None
                # CGAR plan (needs some real history in the window)
                plan = None
                if self.planner is not None:
                    n_win = (int(kv_cache[0]["local_end_index"].item()) - self._sink_lat_now * self.fs) // self.fs
                    if lo > 0 and n_win > 0:
                        plan = self.planner.plan_chunk(self.dw, lo, CHUNK, self.lat_h // 2, self.lat_w // 2)
                    if plan is not None:
                        self.log.append(dict(plan["meta"], chunk=c))
                    ph_bank.stage_corr(plan)
                # warp evidence K/V
                wl, tok_sel, tok_logw = self._material(lo)
                ph_staged, ph_logw = None, None
                if any(t is not None for t in tok_sel):
                    wl_t = torch.stack(wl, dim=1).to(self.device)
                    tmp = self._block_kv(wl_t, lo, cid)
                    parts, lw_parts = [], []
                    for m, vt in enumerate(tok_sel):
                        if vt is None:
                            continue
                        lw_parts.append(torch.from_numpy(tok_logw[m]))
                        parts.append(torch.from_numpy(vt + m * self.fs))
                    sel = torch.cat(parts).to(self.device)
                    ph_logw = torch.cat(lw_parts).float().to(self.device)
                    n_tok = CHUNK * self.fs
                    ph_staged = [(layer["k"][:, :n_tok][:, sel].clone(), layer["v"][:, :n_tok][:, sel].clone())
                                 for layer in tmp]
                    del tmp
                    torch.cuda.empty_cache()
                    self.log.append({"chunk": c, "ph_tokens": int(sel.numel()),
                                     "ph_vis": round(float(sel.numel()) / n_tok, 3)})
                    # clean source block for CGAR source queries (same RoPE, never in the main context)
                    if self.planner is not None:
                        sl_s = torch.stack([self.src_lat[:, lo + m].float() for m in range(CHUNK)], dim=1)
                        tmp3 = self._block_kv(sl_s, lo, cid)
                        kv_s = [(layer["k"][:, :n_tok].clone(), layer["v"][:, :n_tok].clone()) for layer in tmp3]
                        if srcq_cpu:
                            kv_s = [(k.to("cpu").pin_memory(), v.to("cpu").pin_memory()) for k, v in kv_s]
                        ph_bank._corr_kv_s = kv_s
                        ph_bank._corr_sel_s = torch.arange(n_tok, device=self.device)
                        del tmp3
                        torch.cuda.empty_cache()
                # denoising
                current = noise[:, lo:lo + CHUNK]
                for idx in range(len(self.timesteps)):
                    t = self.timesteps[idx]
                    ph_bank._corr_step = idx
                    if dsg is not None:
                        steps = dsg.get("steps")
                        ph_bank._dsg_skip = bool((dsg.get("skip_last", True) and idx == len(self.timesteps) - 1)
                                                 or (steps is not None and idx not in steps))
                    on = ph_staged is not None and idx in ph_steps
                    ph_bank.stage_kv(ph_staged if on else None, tok_logw=ph_logw if on else None)
                    flow = self._forward(current, t, lo, kv_cache, ctx_id=cid)
                    x0 = self.pipe._convert_flow_pred_to_x0(flow_pred=flow, xt=current, timestep=t,
                                                            scheduler=self.pipe.scheduler)
                    if idx < len(self.timesteps) - 1:
                        eps = torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype)
                        current = self.pipe.scheduler.add_noise(x0, eps, self.timesteps[idx + 1])
                pred_chunks.append(x0)
                ph_bank.stage_kv(None)
                ph_bank.stage_corr(None)
                self._forward(x0, 0.0, lo, kv_cache, ctx_id=cid)      # write the chunk into the KV cache
                logging.info(f"chunk {c + 1}/{n_run} done")
        gen_lat = torch.cat(pred_chunks, dim=1)
        del kv_cache
        self.pipe.model.to("cpu")
        torch.cuda.empty_cache()
        return self.pipe.vae.decode([gen_lat], out_cpu=True)[0]
