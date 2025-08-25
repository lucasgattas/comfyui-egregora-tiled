# egregora_core_tiling.py
from __future__ import annotations
from typing import List, Tuple, Dict
import json, math
import torch

def _ensure_multiple_of_8(v: int) -> int:
    return int(v // 8) * 8

def _even_positions(size: int, tile: int, overlap: int, force_mult8: bool = False) -> List[int]:
    """
    Return left/top positions so that:
      - first tile starts at 0
      - last tile ends exactly at size
      - positions are (roughly) evenly spaced
      - requested overlap ~ tile - stride is respected as closely as possible
    """
    tile = int(tile); overlap = int(max(0, min(overlap, tile - 8)))
    avail = max(0, size - tile)
    if avail == 0:
        return [0]

    # initial stride from user intent
    stride = max(1, tile - overlap)
    # number of steps if we used that stride
    n = int(math.ceil(avail / stride)) + 1
    if n <= 1:
        return [0]

    # recompute an even stride to cover [0, size-tile]
    exact = avail / (n - 1)
    if force_mult8:
        exact = max(8.0, round(exact / 8.0) * 8.0)

    xs = [int(round(i * exact)) for i in range(n)]
    xs[0] = 0
    xs[-1] = size - tile
    # clamp & make strictly non-decreasing
    for i in range(1, n):
        xs[i] = max(xs[i-1], min(size - tile, xs[i]))
    # optional: snap to multiples of 8 except the last one (keeps coverage exact)
    if force_mult8:
        for i in range(n - 1):
            xs[i] = int(round(xs[i] / 8) * 8)
        xs[-1] = size - tile
        for i in range(1, n):
            xs[i] = max(xs[i-1], min(size - tile, xs[i]))
    return xs

def _grid_json(windows: List[Dict[str,int]], W: int, H: int, tw: int, th: int, overlap: int,
               rows: int, cols: int) -> str:
    meta = {
        "canvas_px": {"orig_w": W, "orig_h": H, "padded_w": W, "padded_h": H},
        "tile_width": tw, "tile_height": th, "tile_overlap": overlap,
        "rows": rows, "cols": cols,
        "windows_px": windows,
    }
    return json.dumps(meta)

class EgregoraImageTileSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width":  ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "tile_height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "overlap":     ("INT", {"default": 64,  "min": 0,  "max": 1024, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("tiles_batch", "grid_json")
    FUNCTION = "split"
    CATEGORY = "Egregora/Tiled"

    def split(self, image: torch.Tensor, tile_width: int = 768, tile_height: int = 768, overlap: int = 64):
        assert image.dim() == 4, "IMAGE is [B,H,W,C]"
        B, H, W, C = image.shape
        img = image[:1] if B != 1 else image

        tw = _ensure_multiple_of_8(max(8, int(tile_width)))
        th = _ensure_multiple_of_8(max(8, int(tile_height)))
        overlap = max(0, min(int(overlap), tw - 8, th - 8))

        xs = _even_positions(W, tw, overlap, force_mult8=False)
        ys = _even_positions(H, th, overlap, force_mult8=False)

        windows: List[Dict[str,int]] = []
        tiles = []
        for y in ys:
            for x in xs:
                windows.append({"x": x, "y": y, "w": tw, "h": th})
                tiles.append(img[:, y:y+th, x:x+tw, :])

        tiles_batch = torch.cat(tiles, dim=0) if tiles else img.new_zeros((0, th, tw, C))
        grid = _grid_json(windows, W, H, tw, th, overlap, rows=len(ys), cols=len(xs))
        return tiles_batch, grid

NODE_CLASS_MAPPINGS = {"EgregoraImageTileSplit": EgregoraImageTileSplit}
NODE_DISPLAY_NAME_MAPPINGS = {"EgregoraImageTileSplit": "Image Tile Split (Egregora)"}
