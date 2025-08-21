# egregora_tiled_regional_prompt.py
import math, re
import torch
import json
from comfy.sd import CLIP

# ---- Helper Functions ----
def _to_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def _normalize_prompts(prompt_like, total_tiles):
    if isinstance(prompt_like, list):
        prompts = [str(p) for p in prompt_like if str(p).strip()]
    else:
        s = "" if prompt_like is None else str(prompt_like)
        if "\n" in s:
            raw = s.split("\n")
        else:
            raw = s.split(",")
        prompts = [p.strip() for p in raw if p.strip()]
    if not prompts:
        prompts = [""] * total_tiles
    if len(prompts) < total_tiles:
        prompts += [prompts[-1]] * (total_tiles - len(prompts))
    else:
        prompts = prompts[:total_tiles]
    return prompts

def _sanitize(text, blacklist_words):
    if not blacklist_words:
        return text
    if isinstance(blacklist_words, str):
        bl = [w.strip() for w in blacklist_words.split(",") if w.strip()]
    else:
        bl = [str(w).strip() for w in blacklist_words if str(w).strip()]
    if not bl:
        return text
    pattern = r"\b(" + "|".join([re.escape(t) for t in bl]) + r")\b"
    cleaned = re.sub(pattern, "", text or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s*,\s*,", ", ", cleaned)
    return cleaned.strip().strip(", ")

def _combine(global_text, local_text, mode):
    gt = (global_text or "").strip()
    lt = (local_text or "").strip()
    if mode == "replace":
        return gt if gt else lt
    if mode == "append":
        return (lt + (", " if lt and gt else "") + gt) if (lt or gt) else ""
    return (gt + (", " if lt and gt else "") + lt) if (lt or gt) else ""

def _unpack_grid(s):
    if not isinstance(s, str) or not s:
        raise ValueError("grid_json is empty or not a string")
    try:
        g = json.loads(s)
    except json.JSONDecodeError:
        raise ValueError("Invalid grid_json format")
        
    rows = _to_int(g.get("rows", 0))
    cols = _to_int(g.get("cols", 0))
    tile_h = _to_int(g.get("tile_h", 0))
    tile_w = _to_int(g.get("tile_w", 0))
    overlap = _to_int(g.get("overlap", 0))
    origins = [tuple(x) for x in g.get("origins", [])]
    return rows, cols, tile_h, tile_w, overlap, origins

# ---- Node: Regional Prompt Builder ----
class EgregoraTiledRegionalPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "grid_json": ("STRING", {"forceInput": True}),
                "prompt_list": ("STRING", {"multiline": True, "forceInput": True}),
                "strength_pos": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "negative_prompt_list": ("STRING", {"multiline": True, "default": ""}),
                "strength_neg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "global_positive": ("STRING", {"multiline": True, "default": ""}),
                "global_negative": ("STRING", {"multiline": True, "default": ""}),
                "combine_mode": (["prepend", "append", "replace"], {"default": "prepend"}),
                "blacklist_words": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "make_both"
    CATEGORY = "Egregora/Tiled"

    def make_both(
        self, clip, grid_json, prompt_list, strength_pos,
        negative_prompt_list="", strength_neg=1.0,
        global_positive="", global_negative="", combine_mode="prepend",
        blacklist_words=""
    ):
        try:
            rows, cols, tile_h, tile_w, _, origins = _unpack_grid(grid_json)
        except ValueError as e:
            raise ValueError(f"Invalid grid_json: {e}")

        total_tiles = len(origins)
        pos_prompts = _normalize_prompts(prompt_list, total_tiles)
        neg_prompts = _normalize_prompts(negative_prompt_list, total_tiles)
        
        pos_conds, neg_conds = [], []
        
        for i in range(total_tiles):
            y, x = origins[i]
            
            p = _sanitize(pos_prompts[i], blacklist_words)
            p = _combine(global_positive, p, combine_mode)
            
            n = _sanitize(neg_prompts[i], blacklist_words)
            n = _combine(global_negative, n, combine_mode if global_negative else "append")
            
            cond_pos, pooled_pos = clip.encode_from_tokens(clip.tokenize(p), return_pooled=True)
            cond_neg, pooled_neg = clip.encode_from_tokens(clip.tokenize(n), return_pooled=True)
            
            pos_conds.append([cond_pos, {
                "pooled_output": pooled_pos,
                "x": x, "y": y, "w": tile_w, "h": tile_h,
                "strength": float(strength_pos),
            }])
            
            neg_conds.append([cond_neg, {
                "pooled_output": pooled_neg,
                "x": x, "y": y, "w": tile_w, "h": tile_h,
                "strength": float(strength_neg),
            }])
            
        return (pos_conds, neg_conds)

NODE_CLASS_MAPPINGS = {
    "EgregoraTiledRegionalPrompt": EgregoraTiledRegionalPrompt,
}
