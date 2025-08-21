# egregora_tile_split_merge.py
# Split nodes + grid_json-driven Tiled VAE decode (pixel-space blending).
# - Seam-free: distance-to-edge smoothstep + exact normalization
# - No outer black border: disables feather on tiles that touch the canvas edge
# - SDXL/SD1/2 friendly (scale inferred via VAE.decode probe)
# - Batch-safe

import json
import torch
import torch.nn.functional as F


# ----------------------------- helpers ---------------------------------

def _to_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default

def _grid_fit_inside(H, W, tile_h, tile_w, overlap):
    step_y = max(1, tile_h - overlap)
    step_x = max(1, tile_w - overlap)

    last_y = max(0, H - tile_h)
    last_x = max(0, W - tile_w)

    ys = list(range(0, last_y, step_y))
    xs = list(range(0, last_x, step_x))

    if last_y > 0 and (not ys or ys[-1] != last_y):
        ys.append(last_y)
    if last_x > 0 and (not xs or xs[-1] != last_x):
        xs.append(last_x)

    if not ys: ys.append(0)
    if not xs: xs.append(0)

    rows, cols = len(ys), len(xs)
    origins = [(y, x) for y in ys for x in xs]
    return rows, cols, origins

def _order_spiral(origins, rows, cols):
    if not origins or rows == 0 or cols == 0: return []
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
    return [origins[i] for i in idxs if i < len(origins)]

def _pack_grid(rows, cols, tile_h, tile_w, overlap, origins, H=None, W=None):
    return json.dumps({
        "rows": rows, "cols": cols,
        "tile_h": tile_h, "tile_w": tile_w,
        "overlap": overlap, "origins": origins,
        "H": H, "W": W
    })

def _unpack_grid(s):
    g = json.loads(s)
    return (
        _to_int(g.get("rows")),
        _to_int(g.get("cols")),
        _to_int(g.get("tile_h")),
        _to_int(g.get("tile_w")),
        _to_int(g.get("overlap")),
        [tuple(x) for x in g.get("origins", [])],
        _to_int(g.get("H", 0)),
        _to_int(g.get("W", 0)),
    )

def _smoothstep(t):
    # cubic smoothstep: 3t^2 - 2t^3 (expects t in [0,1])
    return t * t * (3.0 - 2.0 * t)


# ------------------------------ NODES -----------------------------------

class EgregoraImageTileSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_width": ("INT", {"default": 512, "min": 64, "step": 16}),
                "tile_height": ("INT", {"default": 512, "min": 64, "step": 16}),
                "overlap": ("INT", {"default": 128, "min": 0, "step": 4}),
                "order": (["row_major", "spiral"],),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("tiles_batch", "grid_json",)
    FUNCTION = "run"
    CATEGORY = "Egregora/Tiled"

    def run(self, image, tile_width, tile_height, overlap, order):
        _, H, W, _ = image.shape
        rows, cols, origins = _grid_fit_inside(H, W, tile_height, tile_width, overlap)
        if order == "spiral":
            origins = _order_spiral(origins, rows, cols)
        if not origins:
            raise ValueError("Image Tile Split: No tiles produced — check sizes.")
        tiles = [image[:, y:y+tile_height, x:x+tile_width, :] for (y, x) in origins]
        return (torch.cat(tiles, dim=0), _pack_grid(rows, cols, tile_height, tile_width, overlap, origins, H, W),)


class EgregoraLatentTileSplit:
    """
    Border-aware latent split: edge-replicate padding so each latent tile has
    plausible context. Output tiles are equal-sized: (B*num_tiles,C,th_l,tw_l).
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "grid_json": ("STRING", {"forceInput": True})}}
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("tiles_batch", "grid_json")
    FUNCTION = "run"
    CATEGORY = "Egregora/Tiled"

    def run(self, latent, grid_json):
        rows, cols, th, tw, overlap, origins, H_img, W_img = _unpack_grid(grid_json)
        if not origins:
            raise ValueError("Latent Tile Split: empty grid/origins.")
        lat = latent["samples"]  # (B,C,H_l,W_l)
        B, C, H_l, W_l = lat.shape

        if H_img <= 0 or W_img <= 0:
            H_img = max(y for y, _ in origins) + th
            W_img = max(x for _, x in origins) + tw

        # infer integer scale f from global ratio (SDXL≈4, SD1/2≈8)
        f = max(1, round(H_img / max(1, H_l)))
        for cand in (8, 4, 2, 1, 16):
            if abs(f - cand) <= 1:
                f = cand
                break

        th_l = (th + f - 1) // f
        tw_l = (tw + f - 1) // f

        tiles = []
        for (y, x) in origins:
            y0, x0 = y // f, x // f
            y1, x1 = y0 + th_l, x0 + tw_l

            cy0, cx0 = max(0, y0), max(0, x0)
            cy1, cx1 = min(H_l, y1), min(W_l, x1)

            src = lat[:, :, cy0:cy1, cx0:cx1]  # (B,C,hc,wc)
            hc, wc = cy1 - cy0, cx1 - cx0
            if hc != th_l or wc != tw_l:
                # Edge-replicate to (th_l, tw_l)
                pad_right  = max(0, tw_l - wc)
                pad_bottom = max(0, th_l - hc)
                src = F.pad(src, (0, pad_right, 0, pad_bottom), mode="replicate")
            tiles.append(src)

        tiles_batch = torch.cat(tiles, dim=0)
        return ({"samples": tiles_batch}, grid_json)


class EgregoraVAEDecodeFromTiles:
    """
    Grid-guided Tiled VAE decode in *pixel space* with exact normalization.
    - Uses distance-to-edge smoothstep weights (seam-free).
    - Disables feather on sides that touch the outer canvas (no black border).
    Inputs:
      latents_list: {'samples': (B*num_tiles,C,th_l,tw_l)}
      grid_json: STRING from split
      vae: VAE
      feather: INT (pixels). 0 => use grid overlap
    Output:
      IMAGE (B,H,W,C)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents_list": ("LATENT",),
                "grid_json": ("STRING", {"forceInput": True}),
                "vae": ("VAE",),
                "feather": ("INT", {"default": 0, "min": 0, "step": 4}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Egregora/Tiled"

    def run(self, latents_list, grid_json, vae, feather):
        tiles_lat = latents_list["samples"]  # (B*num_tiles, C, th_l, tw_l)
        total, C, th_l, tw_l = tiles_lat.shape

        rows, cols, th_px, tw_px, overlap_px, origins, H_img, W_img = _unpack_grid(grid_json)
        num_tiles = len(origins)
        if num_tiles == 0:
            z = tiles_lat.new_zeros(1, 1, 1, 3)  # dummy IMAGE
            return (z,)

        # Batch reshape
        B = max(1, total // num_tiles)
        if B * num_tiles != total:
            B = 1
            tiles_lat = tiles_lat[:num_tiles]
        tiles_lat = tiles_lat.view(B, num_tiles, C, th_l, tw_l)

        # --- Decode ONE tile to learn decoded tile size (Hp,Wp) ---
        with torch.no_grad():
            probe_img = vae.decode(tiles_lat[:, 0, :, :, :])   # IMAGE: (B, Hp, Wp, 3)
        probe = probe_img.movedim(-1, 1)                       # (B, 3, Hp, Wp)
        _, _, Hp, Wp = probe.shape

        # Canvas in pixel space
        if H_img <= 0 or W_img <= 0:
            H_img = max(y for y, _ in origins) + th_px
            W_img = max(x for _, x in origins) + tw_px
        H, W = H_img, W_img

        device = tiles_lat.device
        dtype = probe.dtype  # VAE output dtype

        canvas = torch.zeros((B, 3, H, W), dtype=dtype, device=device)
        weight = torch.zeros((B, 1, H, W), dtype=dtype, device=device)

        # Default feather := grid overlap (pixels)
        use_feather = feather if feather > 0 else overlap_px
        ov = max(1, use_feather)

        # Base distance arrays for a full tile (used then adjusted per-edge)
        base_y = torch.arange(Hp, device=device, dtype=dtype)
        base_x = torch.arange(Wp, device=device, dtype=dtype)
        dist_y = torch.minimum(base_y, (Hp - 1) - base_y)  # distance to nearest horizontal edge
        dist_x = torch.minimum(base_x, (Wp - 1) - base_x)  # distance to nearest vertical edge

        # --- Paste & blend each decoded tile at (y,x) ---
        for t, (y_px, x_px) in enumerate(origins):
            with torch.no_grad():
                rgb_img = vae.decode(tiles_lat[:, t, :, :, :])  # (B, Hp, Wp, 3)
            rgb = rgb_img.movedim(-1, 1)                        # (B, 3, Hp, Wp)

            # Build weight window; disable taper on borders that touch canvas edges
            wy = dist_y.clone()
            wx = dist_x.clone()

            # touching top edge -> no taper on top band
            if y_px <= 0:
                wy[:ov] = wy[ov]
            # touching bottom edge
            if y_px + Hp >= H:
                wy[-ov:] = wy[-ov-1]
            # touching left edge
            if x_px <= 0:
                wx[:ov] = wx[ov]
            # touching right edge
            if x_px + Wp >= W:
                wx[-ov:] = wx[-ov-1]

            ty = torch.clamp(wy / ov, 0.0, 1.0)
            tx = torch.clamp(wx / ov, 0.0, 1.0)
            w2d = torch.ger(_smoothstep(ty), _smoothstep(tx)).view(1, 1, Hp, Wp)  # (1,1,Hp,Wp)

            # clamp paste region to canvas bounds (safety for odd sizes)
            y0, x0 = y_px, x_px
            y1, x1 = y0 + Hp, x0 + Wp

            cy0, cx0 = max(0, y0), max(0, x0)
            cy1, cx1 = min(H, y1), min(W, x1)

            ty0, tx0 = cy0 - y0, cx0 - x0
            ty1, tx1 = ty0 + (cy1 - cy0), tx0 + (cx1 - cx0)

            patch  = rgb[:, :, ty0:ty1, tx0:tx1]
            b_mask = w2d[:, :, ty0:ty1, tx0:tx1]

            canvas[:, :, cy0:cy1, cx0:cx1] += patch * b_mask
            weight[:, :, cy0:cy1, cx0:cx1] += b_mask

        weight = torch.clamp(weight, min=1e-6)
        merged = canvas / weight  # (B,3,H,W)

        # Convert back to Comfy IMAGE (B,H,W,C) in 0..1
        out = torch.clamp(merged.movedim(1, -1), 0.0, 1.0)
        return (out,)


# ------------------------ ComfyUI registration --------------------------

NODE_CLASS_MAPPINGS = {
    "EgregoraImageTileSplit": EgregoraImageTileSplit,
    "EgregoraLatentTileSplit": EgregoraLatentTileSplit,
    "EgregoraVAEDecodeFromTiles": EgregoraVAEDecodeFromTiles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EgregoraImageTileSplit": "Image Tile Split (Egregora)",
    "EgregoraLatentTileSplit": "Latent Tile Split (Egregora)",
    "EgregoraVAEDecodeFromTiles": "VAE Decode From Tiles (Egregora)",
}
