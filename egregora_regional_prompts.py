# egregora_regional_prompts.py
from __future__ import annotations
import json, re
from typing import Any, List
import torch

# ---------------- helpers ----------------

def _clamp01(v: float) -> float:
    v = float(v)
    if v < 0.0: return 0.0
    if v > 1.0: return 1.0
    return v

def _clean_words(s: str, blacklist: List[str]) -> str:
    """Remove blacklist words (whole words, case-insensitive) and tidy spaces."""
    if not blacklist: return s
    out = s
    for w in blacklist:
        w = w.strip()
        if not w: continue
        out = re.sub(rf"\b{re.escape(w)}\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,.;:").strip()
    return out

def _ensure_list(obj) -> List[str]:
    """Accept JSON string / list / scalar and return a list[str]."""
    if obj is None: return []
    if isinstance(obj, list): return [str(x) for x in obj]
    try:
        return [str(x) for x in json.loads(obj)]
    except Exception:
        return [str(obj)]

def _encode_text(clip, text: str):
    """
    Encode with the model's CLIP. Returns (token_embeddings, pooled_output).
    SDXL consumers expect both; Comfy’s CLIP exposes encode_from_tokens(return_pooled=True).
    """
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return cond, pooled

# ---------------- ERP node ----------------

class EgregoraRegionalPrompts:
    """
    Egregora Regional Prompts (ERP) — minimal, intuitive mixing for MoD.

    Positive = weighted mix of TWO branches:
      • BASE   branch: tokens/pooled from `base_positive`
      • TILES  branch: tokens/pooled from concatenated, cleaned `tile_prompts_json`

    Weights are normalized so `w_base + w_tile = 1`.

    SPECIAL CASE (true unconditional):
      If tile_strength == 0 and base_strength == 0, ERP emits
        - positive_cond = encode("")
        - negative_cond = encode("")
        - per-tile pooled = all-zero vectors
      so CFG cancels and sampling is genuinely unconditional.

    Negative (non-unconditional cases) = single branch from `base_negative`.
    Also emits `tile_pooled_json`: per-tile pooled vectors (mixed by the same normalized weights)
    for the MoD wrapper to use per tile.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "grid_json": ("STRING", {"multiline": True}),                    # from your tiler
                "tile_prompts_json": ("STRING", {"multiline": True, "default": ""}),
                # Two sliders only (cap at 1.0)
                "tile_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "base_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "base_positive": ("STRING", {"multiline": True, "default": ""}),
                "base_negative": ("STRING", {"multiline": True, "default": ""}),
                "blacklist": ("STRING", {"multiline": True, "default": ""}),    # optional cleaner for Florence2 lists
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive_cond", "negative_cond", "tile_pooled_json")
    FUNCTION = "build"
    CATEGORY = "Egregora/Prompts"

    # --- small helper to count tiles from grid_json ---
    def _tile_count(self, s: str) -> int:
        try:
            g = json.loads(s or "{}")
            return int(len(g.get("windows_px", [])))
        except Exception:
            return 0

    # --- unconditional helper ---
    def _make_uncond(self, clip, n_tiles: int) -> tuple:
        """Return unconditional positive/negative conds and zero per-tile pooled vectors."""
        tokens_empty, pooled_empty = _encode_text(clip, "")
        positive_cond = [[tokens_empty, {"pooled_output": pooled_empty}]]
        negative_cond = [[tokens_empty, {"pooled_output": pooled_empty}]]
        pooled_dim = int(pooled_empty.numel()) if isinstance(pooled_empty, torch.Tensor) else 0
        zeros = [0.0] * pooled_dim
        tile_pooled_json = json.dumps({
            "pooled_dim": pooled_dim,
            "tiles": [zeros[:] for _ in range(max(0, n_tiles))],
            "texts": [""] * max(0, n_tiles),   # keep schema identical even in uncond mode
        })
        return positive_cond, negative_cond, tile_pooled_json

    # --- main ---
    def build(self,
              clip,
              grid_json,
              tile_prompts_json,
              tile_strength=0.6,
              base_strength=0.4,
              base_positive="",
              base_negative="",
              blacklist=""):

        # Clamp and check unconditional corner case
        w_tile = _clamp01(tile_strength)
        w_base = _clamp01(base_strength)
        n_tiles = self._tile_count(grid_json)

        if (w_tile + w_base) < 1e-6:
            # TRUE UNCONDITIONAL: P = U = encode(""), per-tile pooled = zero vectors
            return self._make_uncond(clip, n_tiles)

        # Normalize weights so w_base + w_tile = 1
        s = w_tile + w_base
        w_tile_n = w_tile / s
        w_base_n = w_base / s

        # Parse / clean tile prompt list (preserve alignment/length = n_tiles)
        raw_list = _ensure_list(tile_prompts_json)
        bl = [t.strip() for t in re.split(r"[,\n]+", blacklist or "") if t.strip()]
        tile_texts = (raw_list + [""] * n_tiles)[:n_tiles] if n_tiles > 0 else []

        # Cleaned-but-aligned list for JSON output (no dedup so indexes stay stable)
        cleaned_by_tile: List[str] = [ _clean_words(t, bl) for t in tile_texts ]

        # Build texts for the two branches (keep dedup only for the global "tiles branch" string)
        seen = set()
        clean_tiles_for_global: List[str] = []
        for t in cleaned_by_tile:
            key = t.lower()
            if not key: 
                continue
            if key in seen:
                continue
            seen.add(key)
            clean_tiles_for_global.append(t)

        base_text  = (base_positive or "").strip()
        tiles_text = " ".join(clean_tiles_for_global).strip()

        # Encode branches (tokens + pooled)
        tokens_base,  pooled_base_pos   = _encode_text(clip, base_text)
        tokens_tiles, pooled_tiles_glob = _encode_text(clip, tiles_text)

        # Positive conditioning: two entries with normalized weights
        positive_cond = []
        if w_base_n > 0.0:
            positive_cond.append([tokens_base,  {"pooled_output": pooled_base_pos,   "weight": float(w_base_n)}])
        if tiles_text and w_tile_n > 0.0:
            positive_cond.append([tokens_tiles, {"pooled_output": pooled_tiles_glob, "weight": float(w_tile_n)}])

        # Negative = single branch (CFG controls intensity)
        tokens_neg, pooled_neg = _encode_text(clip, base_negative or "")
        negative_cond = [[tokens_neg, {"pooled_output": pooled_neg}]]

        # Per-tile pooled vectors for MoD (same normalized weights)
        pooled_list: List[List[float]] = []
        for i in range(n_tiles):
            t_clean = cleaned_by_tile[i] if i < len(cleaned_by_tile) else ""
            if t_clean:
                _, pooled_tile = _encode_text(clip, t_clean)
                mixed = pooled_base_pos * float(w_base_n) + pooled_tile * float(w_tile_n)
            else:
                mixed = pooled_base_pos * float(w_base_n)
            pooled_list.append(mixed.detach().float().view(-1).tolist())

        # --- SURGICAL ADDITION: include per-tile texts so MoD can set per-tile tokens ---
        tile_pooled_json = json.dumps({
            "pooled_dim": int(pooled_base_pos.numel()) if isinstance(pooled_base_pos, torch.Tensor) else None,
            "tiles": pooled_list,
            "texts": cleaned_by_tile,   # <— aligned 1:1 with tiles
        })

        return (positive_cond, negative_cond, tile_pooled_json)


NODE_CLASS_MAPPINGS = {"EgregoraRegionalPrompts": EgregoraRegionalPrompts}
NODE_DISPLAY_NAME_MAPPINGS = {"EgregoraRegionalPrompts": "Egregora Regional Prompts (ERP)"}
