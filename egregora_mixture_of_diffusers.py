from __future__ import annotations
import copy, json
import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F


# ---------- utils ----------
def _align_to(v: int, m: int = 8) -> int:
    return ((int(v) + m - 1) // m) * m


def _normalize_weighted_sum(sum_pred: torch.Tensor, weight_sum: torch.Tensor) -> torch.Tensor:
    return sum_pred / torch.clamp(weight_sum, min=1e-6)


def _reflect_pad(tile: torch.Tensor, mult: int = 8) -> Tuple[torch.Tensor, Tuple[int,int,int,int]]:
    """Pad H,W up to multiple of mult using reflect; return (tensor, (top,bottom,left,right))."""
    _, _, h, w = tile.shape
    th = _align_to(h, mult); tw = _align_to(w, mult)
    ph = max(0, th - h); pw = max(0, tw - w)
    if ph == pw == 0:
        return tile, (0, 0, 0, 0)
    top = ph // 2; bottom = ph - top
    left = pw // 2; right = pw - left
    return F.pad(tile, (left, right, top, bottom), mode="reflect"), (top, bottom, left, right)


def _unpad(t: torch.Tensor, pads: Tuple[int, int, int, int]) -> torch.Tensor:
    top, bottom, left, right = pads
    if top == bottom == left == right == 0:
        return t
    return t[:, :, top:t.shape[-2]-bottom, left:t.shape[-1]-right]


def _build_cosine_weight_sided(h: int, w: int, device, dtype,
                               ov_left: float, ov_right: float, ov_top: float, ov_bottom: float,
                               eps: float = 1e-4) -> torch.Tensor:
    """Edge-aware raised-cosine (Hann) that only tapers on sides with neighbors.
    Returns (1,1,h,w); never exactly zero inside canvas (floor eps).
    """
    h = int(h); w = int(w)
    if h <= 0 or w <= 0:
        return torch.ones((1,1,max(1,h),max(1,w)), device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype).view(1,1,1,w)
    ys = torch.arange(h, device=device, dtype=dtype).view(1,1,h,1)

    def ramp(dist, ov):
        if ov <= 0:
            return torch.ones_like(dist, device=device, dtype=dtype)
        u = torch.clamp(dist / float(max(1.0, ov)), 0, 1)
        return 0.5 * (1 - torch.cos(u * math.pi))  # 0 at edge → 1 in interior

    wx_left  = ramp(xs,               ov_left)
    wx_right = ramp((w - 1) - xs,     ov_right)
    wy_top   = ramp(ys,               ov_top)
    wy_bot   = ramp((h - 1) - ys,     ov_bottom)

    wx = torch.minimum(wx_left, wx_right)
    wy = torch.minimum(wy_top,  wy_bot)
    return (wx * wy).clamp_min(eps)


def _ensure_size_opts(opts: Dict[str,Any], w: int, h: int) -> Dict[str,Any]:
    topts = copy.deepcopy(opts or {})
    size = copy.deepcopy(topts.get("size", {}) or {})
    size["width"] = int(w); size["height"] = int(h)
    topts["size"] = size
    return topts


def _crop_and_weight_control_scaled(obj: Any, ys, ye, xs, xe, H_lat, W_lat, device, dtype, frac_x, frac_y):
    """Scale latent indices to each control tensor's resolution and weight with sqrt(Gaussian)."""
    if isinstance(obj, torch.Tensor):
        _, _, Hs, Ws = obj.shape
        xs_s = int(round(xs * (Ws / max(1, W_lat))))
        xe_s = int(round(xe * (Ws / max(1, W_lat))))
        ys_s = int(round(ys * (Hs / max(1, H_lat))))
        ye_s = int(round(ye * (Hs / max(1, H_lat))))
        xs_s = max(0, min(xs_s, Ws - 1)); xe_s = max(xs_s + 1, min(xe_s, Ws))
        ys_s = max(0, min(ys_s, Hs - 1)); ye_s = max(ys_s + 1, min(ye_s, Hs))
        cropped = obj[:, :, ys_s:ye_s, xs_s:xe_s]
        # simple center-weight to favor interior
        h_s, w_s = cropped.shape[-2], cropped.shape[-1]
        sx_s = max(1.0, frac_x * w_s); sy_s = max(1.0, frac_y * h_s)
        x = torch.arange(w_s, device=device, dtype=dtype) - (w_s - 1) / 2.0
        y = torch.arange(h_s, device=device, dtype=dtype) - (h_s - 1) / 2.0
        gx = torch.exp(-0.5 * (x / sx_s) ** 2); gy = torch.exp(-0.5 * (y / sy_s) ** 2)
        m = (gy.view(1,1,h_s,1) * gx.view(1,1,1,w_s)).sqrt()
        return cropped * m

    if isinstance(obj, dict):
        return {k: _crop_and_weight_control_scaled(v, ys, ye, xs, xe, H_lat, W_lat, device, dtype, frac_x, frac_y) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_crop_and_weight_control_scaled(v, ys, ye, xs, xe, H_lat, W_lat, device, dtype, frac_x, frac_y) for v in obj]
    return obj


# ---------- helpers for neighbor detection ----------
def _infer_overlaps_px(windows_px: List[Dict[str,int]]) -> Tuple[int,int,int,int]:
    """Infer (tile_w, tile_h, ovx, ovy) in *pixel* units from a windows list."""
    xs = [w["x"] for w in windows_px]; ys = [w["y"] for w in windows_px]
    ws = [w["w"] for w in windows_px]; hs = [w["h"] for w in windows_px]
    tw = min(ws) if ws else 0; th = min(hs) if hs else 0
    xs_sorted = sorted(xs); ys_sorted = sorted(ys)
    dx = min([xs_sorted[i+1]-xs_sorted[i] for i in range(len(xs_sorted)-1)] or [tw])
    dy = min([ys_sorted[i+1]-ys_sorted[i] for i in range(len(ys_sorted)-1)] or [th])
    ovx = max(0, tw - dx); ovy = max(0, th - dy)
    return tw, th, ovx, ovy


def _has_neighbor_px(windows_px: List[Dict[str,int]], i: int) -> Tuple[bool,bool,bool,bool]:
    """Check whether tile i has touching neighbor(s) on each side with span overlap."""
    x = windows_px[i]["x"]; y = windows_px[i]["y"]
    w = windows_px[i]["w"]; h = windows_px[i]["h"]

    def overlap(a1,a2,b1,b2): return not (b2 <= a1 or b1 >= a2)

    left = right = top = bottom = False
    for j, wj in enumerate(windows_px):
        if j == i: continue
        x2, y2, w2, h2 = wj["x"], wj["y"], wj["w"], wj["h"]
        if x2 + w2 == x and overlap(y, y+h, y2, y2+h2): left = True
        if x2 == x + w and overlap(y, y+h, y2, y2+h2): right = True
        if y2 + h2 == y and overlap(x, x+w, x2, x2+w2): top = True
        if y2 == y + h and overlap(x, x+w, x2, x2+w2): bottom = True
    return left, right, top, bottom


# ---------- MoD wrapper ----------
class _EgregoraMoDWrapper:
    def __init__(self, windows_px: List[Dict[str,int]], canvas_px: Dict[str,int],
                 tile_pooled: List[List[float]] | None = None,
                 schedule: Dict[str,float] | None = None,
                 tile_texts: List[str] | None = None,
                 clip=None, debug: bool = False):
        self.windows_px = windows_px
        self.canvas_px = canvas_px or {}
        self.tile_pooled = tile_pooled
        self.schedule = schedule or {}
        self.tile_texts = tile_texts or []
        self.clip = clip
        self._tokens_cache: Dict[Tuple[str,str], torch.Tensor] = {}
        self.debug = bool(debug)
        self._tmax = None
        self._tmin_seen = None

    # Comfy will call us as a function
    def __call__(self, model_function, *args, **kwargs):
        # modern dict style
        if len(args) == 1 and isinstance(args[0], dict):
            return self._call_modern(model_function, args[0])
        # kwargs style
        if "input" in kwargs and ("timestep" in kwargs or "t" in kwargs):
            params = {
                "input": kwargs["input"],
                "timestep": kwargs.get("timestep", kwargs.get("t")),
                "c": kwargs.get("c", {}),
                "model_options": kwargs.get("model_options", {}),
                "cond_or_uncond": kwargs.get("cond_or_uncond", None),
            }
            return self._call_modern(model_function, params)
        # legacy positional: (x, t, c, **kwargs)
        if len(args) >= 3 and isinstance(args[0], torch.Tensor):
            x, t, c = args[0], args[1], args[2]
            return self._call_modern(model_function, {"input": x, "timestep": t, "c": c})
        # last resort
        raise TypeError("Unsupported call signature for _EgregoraMoDWrapper")

    # ----- helpers -----
    def _encode_tokens_cached(self, text: str, device, dtype):
        if not text or self.clip is None:
            return None
        key = (text, str(dtype))
        tok = self._tokens_cache.get(key)
        if tok is not None:
            try: return tok.to(device)
            except Exception: return tok
        tokens = self.clip.tokenize(text)
        try:
            cond, _ = self.clip.encode_from_tokens(tokens, return_pooled=True)
        except TypeError:
            out = self.clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = out.get("cond", None)
        if cond is not None:
            try: self._tokens_cache[key] = cond.detach().to("cpu")
            except Exception: self._tokens_cache[key] = cond
            return cond.to(device)
        return None

    def _progress_weights(self, t_cur):
        # linear schedule over first->last timestep
        if self._tmax is None:
            self._tmax = float(t_cur); self._tmin_seen = float(t_cur)
        self._tmin_seen = min(self._tmin_seen, float(t_cur))
        prog = 0.0 if self._tmax <= self._tmin_seen else (self._tmax - float(t_cur)) / max(1e-6, self._tmax - self._tmin_seen)
        ts = self.schedule.get("tile_start", 1.0); te = self.schedule.get("tile_end", 1.0)
        bs = self.schedule.get("base_start", 0.0); be = self.schedule.get("base_end", 0.0)
        tile_w = max(0.0, min(1.0, ts + (te - ts) * prog))
        base_w = max(0.0, min(1.0, bs + (be - bs) * prog))
        w_norm = max(1e-6, tile_w + base_w)
        return tile_w, base_w, w_norm

    # ----- core path -----
    def _call_modern(self, model_function, params: Dict[str,Any]):
        x = params["input"]; t = params["timestep"]; c = params.get("c", {}) or {}
        branch = params.get("cond_or_uncond", None)

        B,C,H,W = x.shape
        device, dtype = x.device, x.dtype

        canvas_w = int(self.canvas_px.get("padded_w", self.canvas_px.get("orig_w", W*8)))
        canvas_h = int(self.canvas_px.get("padded_h", self.canvas_px.get("orig_h", H*8)))

        # latent<->pixel scale
        sx = W / max(1, canvas_w); sy = H / max(1, canvas_h)

        # infer tile sizes and overlaps in px -> convert to latent
        tw_px, th_px, ovx_px, ovy_px = _infer_overlaps_px(self.windows_px)
        tw_lat = max(1.0, float(tw_px) * sx) if tw_px else float(W)
        th_lat = max(1.0, float(th_px) * sy) if th_px else float(H)
        ovx_lat = max(0.0, float(ovx_px) * sx); ovy_lat = max(0.0, float(ovy_px) * sy)

        # context + control-map scaling
        sigma_x = max(0.15 * tw_lat, 0.50 * ovx_lat, 1.0)
        sigma_y = max(0.15 * th_lat, 0.50 * ovy_lat, 1.0)
        frac_x = sigma_x / max(1.0, tw_lat); frac_y = sigma_y / max(1.0, th_lat)

        # optional controls
        precomp_ctrl = c.get("control", None)
        base_ctx = c.get("c_crossattn", None)
        base_topts = c.get("transformer_options", {})

        weight_sum = torch.zeros_like(x)
        sum_pred  = torch.zeros_like(x)

        tile_w, base_w, w_norm = self._progress_weights(t)

        # Detect if *this* model is class-conditional; only then can we pass y
        supports_y = False
        try:
            bound_self = getattr(model_function, "__self__", None)  # BaseModel
            dm = getattr(bound_self, "diffusion_model", None)
            num_classes = getattr(dm, "num_classes", None)
            supports_y = (num_classes is not None)
        except Exception:
            supports_y = False  # safest
        # Strip stray 'y' unless model is class-conditional (avoids SamplerCustom assertion)
        if not supports_y and isinstance(c, dict) and 'y' in c:
            c = {k:v for k,v in c.items() if k != 'y'}

        for i, w in enumerate(self.windows_px):
            xs = int(round(w["x"] * sx)); ys = int(round(w["y"] * sy))
            xe = int(round((w["x"] + w["w"]) * sx)); ye = int(round((w["y"] + w["h"]) * sy))
            xs = max(0, min(xs, W-1)); xe = max(xs+1, min(xe, W))
            ys = max(0, min(ys, H-1)); ye = max(ys+1, min(ye, H))

            # give UNet extra context around the tile to avoid seams
            ctx_x = int(max(8, ovx_lat, tw_lat * 0.12))
            ctx_y = int(max(8, ovy_lat, th_lat * 0.12))
            xs_ctx = max(0, xs - ctx_x); xe_ctx = min(W, xe + ctx_x)
            ys_ctx = max(0, ys - ctx_y); ye_ctx = min(H, ye + ctx_y)
            tile_lat_ctx = x[:, :, ys_ctx:ye_ctx, xs_ctx:xe_ctx]
            tile_lat_padded, pads = _reflect_pad(tile_lat_ctx, mult=8)

            # build per-tile conditioning
            c_tile_ctx = base_ctx
            if (branch == "cond" or branch is None) and hasattr(self, "tile_texts") and self.tile_texts and i < len(self.tile_texts):
                tok = self._encode_tokens_cached(self.tile_texts[i], device, dtype)
                if tok is not None:
                    c_tile_ctx = tok

            # pooled mixing goes into transformer_options
            topts = _ensure_size_opts(base_topts, canvas_w, canvas_h)
            # If SDXL add_time_ids are present, update crop coords for this tile only
            try:
                base_add = topts.get("add_time_ids", None)
                if base_add is not None:
                    if torch.is_tensor(base_add):
                        add = base_add.clone()
                        if add.ndim == 2 and add.shape[0] == 1 and add.shape[1] >= 4:
                            add[:, 2] = float(int(self.windows_px[i]["y"]))
                            add[:, 3] = float(int(self.windows_px[i]["x"]))
                            topts["add_time_ids"] = add
                    elif isinstance(base_add, (list, tuple)) and len(base_add) >= 4:
                        add = list(base_add)
                        add[2] = int(self.windows_px[i]["y"])
                        add[3] = int(self.windows_px[i]["x"])
                        topts["add_time_ids"] = add
            except Exception:
                pass

            tile_vec = None
            if self.tile_pooled and i < len(self.tile_pooled):
                tile_vec = torch.tensor(self.tile_pooled[i], device=device, dtype=dtype).view(1, -1)
            base_vec = topts.get("pooled_output", None)

            if (branch == "cond" or branch is None):
                if tile_vec is not None and base_vec is not None and w_norm > 0:
                    base_t = base_vec if torch.is_tensor(base_vec) else torch.tensor(base_vec, device=device, dtype=dtype).view(1,-1)
                    mixed = (tile_vec * (tile_w / w_norm)) + (base_t * (base_w / w_norm))
                    topts["pooled_output"] = mixed
                elif tile_vec is not None:
                    topts["pooled_output"] = tile_vec

            # control: crop & weight at each scale (scaled indices)
            call_control = None
            if isinstance(precomp_ctrl, (dict, list, torch.Tensor)):
                call_control = _crop_and_weight_control_scaled(precomp_ctrl, ys_ctx, ye_ctx, xs_ctx, xe_ctx, H, W, device, dtype, frac_x, frac_y)

            # ---- safe UNet call: pass only allowed kwargs; pass 'y' only if model needs it ----
            call_kwargs = {
                "c_crossattn": c_tile_ctx,
                "control": call_control,
                "transformer_options": topts,
            }
            if supports_y and "y" in c and c["y"] is not None:
                call_kwargs["y"] = c["y"]

            pred = model_function(tile_lat_padded, t, **call_kwargs)
            pred = _unpad(pred, pads)

            # crop back from context region to the original tile window
            ys0 = ys - ys_ctx; xs0 = xs - xs_ctx
            pred = pred[:, :, ys0:ys0 + (ye - ys), xs0:xs0 + (xe - xs)]

            # neighbor-aware overlap per side (in *latent* px)
            hasL, hasR, hasT, hasB = _has_neighbor_px(self.windows_px, i)
            ovL = ovx_lat if hasL else 0.0
            ovR = ovx_lat if hasR else 0.0
            ovT = ovy_lat if hasT else 0.0
            ovB = ovy_lat if hasB else 0.0

            # if grid reports zero-overlap everywhere, synthesize a small band for interior tiles
            if ovx_lat == 0.0 and ovy_lat == 0.0:
                synth_x = float(max(1.0, 0.12 * (xe - xs)))
                synth_y = float(max(1.0, 0.12 * (ye - ys)))
                ovL = synth_x if hasL else 0.0
                ovR = synth_x if hasR else 0.0
                ovT = synth_y if hasT else 0.0
                ovB = synth_y if hasB else 0.0

            # build edge-aware mask (never zero on border without neighbor)
            w_mask = _build_cosine_weight_sided(pred.shape[-2], pred.shape[-1], device, dtype, ovL, ovR, ovT, ovB)

            sum_pred[:, :, ys:ye, xs:xe] += pred * w_mask
            weight_sum[:, :, ys:ye, xs:xe] += w_mask

        return _normalize_weighted_sum(sum_pred, weight_sum)


# ---------- node ----------
class EgregoraMixtureOfDiffusers:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "grid_json": ("STRING", {"multiline": True}),
            },
            "optional": {
                "tile_pooled_json": ("STRING", {"multiline": True, "default": ""}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "patch"
    CATEGORY = "Egregora/MoD"

    def _parse_grid(self, s: str):
        try:
            g = json.loads(s or "{}")
        except Exception as e:
            print(f"[EgregoraMoD] Invalid grid_json: {e}")
            return [], {}
        return g.get("windows_px", []), g.get("canvas_px", {})

    def _parse_tile_pooled(self, s: str):
        if not s:
            return None, {}, None
        try:
            obj = json.loads(s)
            return obj.get("tiles", None), obj.get("schedule", {}) or {}, obj.get("texts", None)
        except Exception:
            return None, {}, None

    def patch(self, model, clip, grid_json, tile_pooled_json: str = "", debug: bool = False):
        # ComfyUI passes a ModelPatcher for MODEL; we clone and wrap its UNet
        if not hasattr(model, "clone"):
            print("[EgregoraMoD] Provided model is not a ModelPatcher; returning as-is.")
            return (model,)

        windows_px, canvas_px = self._parse_grid(grid_json)
        if not windows_px:
            print("[EgregoraMoD] No windows in grid_json; returning model unchanged.")
            return (model,)

        tile_pooled, schedule, tile_texts = self._parse_tile_pooled(tile_pooled_json)
        patched = model.clone()
        wrapper = _EgregoraMoDWrapper(
            windows_px=windows_px,
            canvas_px=canvas_px,
            tile_pooled=tile_pooled,
            schedule=schedule,
            tile_texts=tile_texts,
            clip=clip,
            debug=bool(debug),
        )
        patched.set_model_unet_function_wrapper(wrapper)
        return (patched,)


# ---- required mappings for ComfyUI loader ----
NODE_CLASS_MAPPINGS = {"EgregoraMixtureOfDiffusers": EgregoraMixtureOfDiffusers}
NODE_DISPLAY_NAME_MAPPINGS = {"EgregoraMixtureOfDiffusers": "Egregora Mixture of Diffusers (MoD)"}
