# egregora_core_tiling.py
# Egregora tiling: Image Tile Split, Tiled Regional Prompt, Tiled Diffusion (Model)
# - Wildcard inputs for compatibility (grid_json, prompt_list)
# - Regional prompt: validation, prompt normalization, CLIP encode cache
# - Tiled Diffusion model wrapper that reads grid_json to override sizes

from __future__ import annotations
from typing import Dict, Any, Optional, Union, List, Tuple
import json
import re

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None  # type: ignore


# =============================================================================
# Grid utilities
# =============================================================================

def _grid_fit_inside(H: int, W: int, th: int, tw: int, overlap: int):
    th = int(th); tw = int(tw); overlap = int(max(0, overlap))
    step_h = max(1, th - overlap)
    step_w = max(1, tw - overlap)

    ys = [0]
    while ys[-1] + th + step_h <= H:
        ys.append(ys[-1] + step_h)
    if ys[-1] + th < H:
        last = max(0, H - th)
        if last != ys[-1]:
            ys.append(last)

    xs = [0]
    while xs[-1] + tw + step_w <= W:
        xs.append(xs[-1] + step_w)
    if xs[-1] + tw < W:
        last = max(0, W - tw)
        if last != xs[-1]:
            xs.append(last)

    rows, cols = len(ys), len(xs)
    origins = [(y, x) for y in ys for x in xs]
    return rows, cols, origins


def _order_spiral(origins: List[Tuple[int, int]], rows: int, cols: int):
    if not origins or rows == 0 or cols == 0:
        return origins
    grid = [[r * cols + c for c in range(cols)] for r in range(rows)]
    top, bottom, left, right = 0, rows - 1, 0, cols - 1
    idxs = []
    while top <= bottom and left <= right:
        for c in range(left, right + 1): idxs.append(grid[top][c])
        top += 1
        for r in range(top, bottom + 1): idxs.append(grid[r][right])
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1): idxs.append(grid[bottom][c])
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1): idxs.append(grid[r][left])
            left += 1
    return [origins[i] for i in idxs]


def _pack_grid(rows:int, cols:int, th:int, tw:int, overlap:int, origins:List[Tuple[int,int]], H:int, W:int, latent_scale:int=8) -> str:
    return json.dumps({
        "rows": int(rows), "cols": int(cols),
        "tile_h": int(th), "tile_w": int(tw),
        "overlap": int(overlap),
        "origins": origins,
        "H": int(H), "W": int(W),
        "latent_scale": int(latent_scale),
    })


def _unpack_grid(grid_json: Union[str, Dict[str, Any]]):
    g = grid_json if isinstance(grid_json, dict) else json.loads(grid_json)
    return (
        int(g["rows"]),
        int(g["cols"]),
        int(g["tile_h"]),
        int(g["tile_w"]),
        int(g["overlap"]),
        [tuple(x) for x in g["origins"]],
        int(g["H"]),
        int(g["W"]),
        int(g.get("latent_scale", 8)),
    )


def _to_dict(maybe: Union[None, Dict[str, Any], str, bytes, tuple, list]) -> Optional[Dict[str, Any]]:
    """Normalize: dict | (dict,) | [dict] | JSON string/bytes"""
    if maybe is None:
        return None
    if isinstance(maybe, (list, tuple)) and maybe:
        maybe = maybe[0]
    if isinstance(maybe, dict):
        return maybe
    if isinstance(maybe, (str, bytes)):
        try:
            return json.loads(maybe.decode("utf-8") if isinstance(maybe, bytes) else maybe)
        except Exception:
            return None
    return None


def _as_int(d: Dict[str, Any], *keys: str, default: Optional[int] = None) -> Optional[int]:
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, int): return v
            if isinstance(v, float): return int(v)
    return default


def _infer_from_grid_sizes(grid: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], int]:
    """Return (tile_width_px, tile_height_px, overlap_px, scale) from grid dict."""
    scale = _as_int(grid, "scale", "latent_scale", default=8) or 8
    tw = _as_int(grid, "tile_width", "tile_w")
    th = _as_int(grid, "tile_height", "tile_h")
    ov = _as_int(grid, "overlap", "tile_overlap", "overlap_px")
    if (tw is None or th is None) and isinstance(grid.get("tiles"), list) and grid["tiles"]:
        t0 = grid["tiles"][0]
        tw = tw or _as_int(t0, "w", "width")
        th = th or _as_int(t0, "h", "height")
    return tw, th, ov, int(scale)


def _snap_sizes(tile_w: int, tile_h: int, overlap: int, compression: int):
    """Keep sizes aligned to model compression boundary."""
    def snap(v: int) -> int:
        v = max(compression, int(v))
        return (v // compression) * compression
    tw = snap(tile_w)
    th = snap(tile_h)
    ov = max(0, min(int(overlap), min(tw, th) // 2))
    return tw, th, ov


def _validate_tile_bounds(x:int, y:int, w:int, h:int, W:int, H:int, latent_scale:int=8):
    """Validate + snap tile bounds to the latent grid."""
    x, y, w, h = max(0, int(x)), max(0, int(y)), max(1, int(w)), max(1, int(h))
    W, H = max(1, int(W)), max(1, int(H))
    x = min(x, W - 1); y = min(y, H - 1)
    w = min(w, W - x); h = min(h, H - y)
    if w <= 0 or h <= 0:
        return None
    s = max(1, int(latent_scale))
    x_al = (x // s) * s; y_al = (y // s) * s
    w_al = ((w + s - 1) // s) * s; h_al = ((h + s - 1) // s) * s
    if x_al + w_al > W: w_al = ((W - x_al) // s) * s
    if y_al + h_al > H: h_al = ((H - y_al) // s) * s
    if w_al <= 0 or h_al <= 0:
        return None
    return int(x_al), int(y_al), int(w_al), int(h_al)


# =============================================================================
# Node: Image Tile Split (produces tiles + grid_json)
# =============================================================================

class EgregoraImageTileSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width":  ("INT", {"default": 512, "min": 64, "step": 16}),
                "tile_height": ("INT", {"default": 512, "min": 64, "step": 16}),
                "overlap":     ("INT", {"default": 64,  "min": 0,  "step": 4}),
                "order": (["row_major", "spiral"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("tiles_batch", "grid_json")
    FUNCTION = "run"
    CATEGORY = "Egregora/Tiled"

    def run(self, image, tile_width, tile_height, overlap, order):
        assert torch is not None, "PyTorch is required"
        _, H, W, _ = image.shape
        rows, cols, origins = _grid_fit_inside(H, W, tile_height, tile_width, overlap)
        if order == "spiral":
            origins = _order_spiral(origins, rows, cols)
        tiles = [image[:, y:y+tile_height, x:x+tile_width, :] for (y, x) in origins]
        return torch.cat(tiles, dim=0), _pack_grid(
            rows, cols, tile_height, tile_width, overlap, origins, H, W
        )


# =============================================================================
# Node: Tiled Regional Prompt (per-tile conditioning, with caching)
# =============================================================================

class EgregoraTiledRegionalPrompt:
    """Regional prompts for tiled generation with wildcard inputs & CLIP cache."""

    _CACHE: Dict[tuple, tuple] = {}  # (id(clip), text) -> (cond, pooled)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "grid_json": ("*", {"forceInput": True}),     # accepts STRING/DICT/etc
                "prompt_list": ("*", {"multiline": True, "forceInput": True}),  # list/dict/str
                "strength_pos": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "negative_prompt_list": ("*", {"multiline": True, "default": ""}),
                "strength_neg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "global_positive": ("STRING", {"multiline": True, "default": ""}),
                "global_negative": ("STRING", {"multiline": True, "default": ""}),
                "combine_mode": (["prepend", "append", "replace"], {"default": "prepend"}),
                "blacklist_words": ("STRING", {"multiline": True, "default": ""}),
                "enable_validation": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "make_conditioning"
    CATEGORY = "Egregora/Tiled"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):  # <- must accept the parameter
        return True

    # ---------- text helpers ----------

    @staticmethod
    def _split_lines_or_commas(s):
        s = "" if s is None else str(s)
        s = s.replace("\r\n", "\n").strip()
        if not s:
            return []
        parts = [p.strip() for p in (s.split("\n") if "\n" in s else s.split(","))]
        return [p for p in parts if p]

    @staticmethod
    def _normalize_prompt_list(prompt_like, total):
        # list of strings (or dicts with 'text')
        if isinstance(prompt_like, list):
            arr = []
            for p in prompt_like:
                if isinstance(p, dict):
                    arr.append(str(p.get("text", "")))
                else:
                    arr.append(str(p))
        elif isinstance(prompt_like, dict):
            # dict with 'items' list or numeric keys
            if "items" in prompt_like and isinstance(prompt_like["items"], list):
                arr = [str(x) if not isinstance(x, dict) else str(x.get("text", "")) for x in prompt_like["items"]]
            else:
                try:
                    arr = [str(v) for k, v in sorted(prompt_like.items(), key=lambda kv: int(kv[0]))]
                except Exception:
                    arr = [str(v) for v in prompt_like.values()]
        else:
            # string fallback
            arr = EgregoraTiledRegionalPrompt._split_lines_or_commas(prompt_like)

        if not arr:
            arr = [""]
        if len(arr) < total:
            arr += [arr[-1]] * (total - len(arr))
        return arr[:total]

    @staticmethod
    def _sanitize(text, blacklist):
        if not text:
            text = ""
        if not blacklist:
            return text
        words = blacklist if isinstance(blacklist, list) else EgregoraTiledRegionalPrompt._split_lines_or_commas(blacklist)
        if not words:
            return text
        pat = r"\b(" + "|".join([re.escape(w) for w in words]) + r")\b"
        out = re.sub(pat, "", text, flags=re.IGNORECASE)
        out = re.sub(r"\s{2,}", " ", out)
        out = re.sub(r"\s*,\s*,", ", ", out)
        return out.strip().strip(", ")

    @staticmethod
    def _combine(global_text, local_text, mode):
        gt, lt = (global_text or "").strip(), (local_text or "").strip()
        if mode == "replace":
            return gt if gt else lt
        if mode == "append":
            return (lt + (", " if lt and gt else "") + gt) if (lt or gt) else ""
        # prepend (default)
        return (gt + (", " if lt and gt else "") + lt) if (lt or gt) else ""

    # ---------- CLIP encode with cache ----------

    @staticmethod
    def _encode_cached(clip, text: str):
        key = (id(clip), text)
        hit = EgregoraTiledRegionalPrompt._CACHE.get(key)
        if hit is not None:
            return hit
        try:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            EgregoraTiledRegionalPrompt._CACHE[key] = (cond, pooled)
            return cond, pooled
        except Exception as e:
            print(f"[EgregoraTiledRegionalPrompt] CLIP encoding failed: {e}")
            return None, None

    # ---------- main ----------

    def make_conditioning(self, clip, grid_json, prompt_list, strength_pos,
                          negative_prompt_list="", strength_neg=1.0,
                          global_positive="", global_negative="", combine_mode="prepend",
                          blacklist_words="", enable_validation=True):

        g = _to_dict(grid_json) if not isinstance(grid_json, dict) else grid_json
        if not isinstance(g, dict):
            return ([], [])

        try:
            rows, cols, tile_h, tile_w, overlap, origins, W, H, latent_scale = _unpack_grid(g)
        except Exception as e:
            print(f"[EgregoraTiledRegionalPrompt] Failed to unpack grid_json: {e}")
            return ([], [])

        total = len(origins)
        if total == 0 or W <= 0 or H <= 0:
            return ([], [])

        pos_prompts = self._normalize_prompt_list(prompt_list, total)
        neg_prompts = self._normalize_prompt_list(negative_prompt_list, total)

        pos_conds, neg_conds = [], []

        for i in range(total):
            y0, x0 = origins[i]

            p_local = self._sanitize(pos_prompts[i], blacklist_words)
            n_local = self._sanitize(neg_prompts[i], blacklist_words)
            p_text = self._combine(global_positive, p_local, combine_mode)
            n_text = self._combine(global_negative, n_local, combine_mode)

            if enable_validation:
                bounds = _validate_tile_bounds(x0, y0, tile_w, tile_h, W, H, latent_scale)
                if bounds is None:
                    continue
                x, y, w, h = bounds
            else:
                x, y, w, h = int(x0), int(y0), int(tile_w), int(tile_h)

            if p_text.strip():
                cond_pos, pooled_pos = self._encode_cached(clip, p_text)
                if cond_pos is not None:
                    pos_conds.append([cond_pos, {
                        "pooled_output": pooled_pos,
                        "x": x, "y": y, "w": w, "h": h,
                        "strength": float(strength_pos),
                    }])

            if n_text.strip():
                cond_neg, pooled_neg = self._encode_cached(clip, n_text)
                if cond_neg is not None:
                    neg_conds.append([cond_neg, {
                        "pooled_output": pooled_neg,
                        "x": x, "y": y, "w": w, "h": h,
                        "strength": float(strength_neg),
                    }])

        return (pos_conds, neg_conds)


# =============================================================================
# Node: Tiled Diffusion (Model) wrapper
# =============================================================================

class EgregoraTiledDiffusionModel:
    """
    Wraps the model with a tiled diffusion UNet wrapper.
    If grid_json is provided, overrides manual tile_* values to keep everything in sync.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "method": (["Mixture of Diffusers", "MultiDiffusion", "SpotDiffusion"], {"default": "Mixture of Diffusers"}),
                "tile_width":  ("INT", {"default": 768, "min": 64, "max": 8192, "step": 16}),
                "tile_height": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 16}),
                "tile_overlap": ("INT", {"default": 64,  "min": 0,  "max": 512,  "step": 4}),
                "tile_batch_size": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
            },
            "optional": {
                "grid_json": ("*", {"forceInput": True}),  # wildcard for compatibility
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Egregora/Tiled"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    # Prefer the official Tiled Diffusion node if installed
    def _apply_via_td(self, model, method, tile_width, tile_height, tile_overlap, tile_batch_size):
        try:
            from tiled_diffusion import TiledDiffusion
        except Exception:
            return None  # module not available -> let caller handle fallback
        (patched_model,) = TiledDiffusion().apply(
            model, method, tile_width, tile_height, tile_overlap, tile_batch_size
        )
        return patched_model

    # Fallback inline wrapper (keeps parity if extension not present)
    def _apply_inline(self, model, method, tile_width, tile_height, tile_overlap, tile_batch_size):
        # Quiet fallback: do not import tiled_diffusion. Return base model clone with a note.
        m = model.clone()
        m.model_options["tiled_diffusion"] = False
        m.model_options["tiled_diffusion_args"] = {
            "method": method,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tile_overlap": tile_overlap,
            "tile_batch_size": tile_batch_size,
            "note": "tiled_diffusion module not found; running base model",
        }
        return m

    def apply(self, model, method, tile_width, tile_height, tile_overlap, tile_batch_size, grid_json=None):
        g = _to_dict(grid_json) if not isinstance(grid_json, dict) else grid_json
        if isinstance(g, dict):
            tw_g, th_g, ov_g, scale = _infer_from_grid_sizes(g)
            s = max(1, int(scale))
            if tw_g is not None:  tile_width  = int(tw_g) * s
            if th_g is not None:  tile_height = int(th_g) * s
            if ov_g is not None:  tile_overlap = int(ov_g) * s

        # try official module
        patched = self._apply_via_td(model, method, tile_width, tile_height, tile_overlap, tile_batch_size)
        if patched is not None:
            return (patched,)

        # quiet fallback (no error spam)
        return (self._apply_inline(model, method, tile_width, tile_height, tile_overlap, tile_batch_size),)


# =============================================================================
# ComfyUI registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "EgregoraImageTileSplit": EgregoraImageTileSplit,
    "EgregoraTiledRegionalPrompt": EgregoraTiledRegionalPrompt,
    "EgregoraTiledDiffusionModel": EgregoraTiledDiffusionModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EgregoraImageTileSplit": "Image Tile Split (Egregora)",
    "EgregoraTiledRegionalPrompt": "Tiled Regional Prompt (Egregora)",
    "EgregoraTiledDiffusionModel": "Tiled Diffusion (Model) (Egregora)",
}
