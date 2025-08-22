# egregora_tiled_model.py
# [Egregora] Tiled Diffusion (Model) — grid_json via wildcard input

from typing import Tuple, Dict, Any, Optional, Union
import json

try:
    import comfy
    from comfy.model_patcher import ModelPatcher
except Exception:
    comfy = None
    ModelPatcher = object  # type: ignore


def _as_int(d: Dict[str, Any], *keys: str, default: Optional[int] = None) -> Optional[int]:
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                return int(v)
    return default

def _infer_from_grid(grid: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int], int]:
    scale = _as_int(grid, "scale", "latent_scale", default=8) or 8
    tw = _as_int(grid, "tile_width", "tile_w")
    th = _as_int(grid, "tile_height", "tile_h")
    ov = _as_int(grid, "overlap", "tile_overlap")
    if (tw is None or th is None) and isinstance(grid.get("tiles"), list) and grid["tiles"]:
        t0 = grid["tiles"][0]
        tw = tw or _as_int(t0, "w", "width")
        th = th or _as_int(t0, "h", "height")
    return tw, th, ov, int(scale)

def _snap_sizes(tile_w: int, tile_h: int, overlap: int, compression: int) -> Tuple[int, int, int]:
    def snap(v: int) -> int:
        v = max(compression, int(v))
        return (v // compression) * compression
    tw = snap(tile_w); th = snap(tile_h)
    ov = max(0, min(int(overlap), min(tw, th) // 2))
    return tw, th, ov

def _to_dict(maybe: Union[None, Dict[str, Any], str, bytes, tuple, list]) -> Optional[Dict[str, Any]]:
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


class EgregoraTiledDiffusionModel:
    """
    Tiled Diffusion model wrapper that accepts ANY input on grid_json.
    When grid_json is connected, it overrides manual tile_* values.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "method": (["Mixture of Diffusers", "MultiDiffusion", "SpotDiffusion"], {"default": "Mixture of Diffusers"}),
                "tile_width":  ("INT", {"default": 768, "min": 64, "max": 8192, "step": 16}),
                "tile_height": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 16}),
                "tile_overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 4}),
                "tile_batch_size": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
            },
            "optional": {
                # Wildcard so it will CONNECT regardless of the producer's datatype.
                "grid_json": ("*", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Egregora/Tiled"

    # Allow any incoming type (required for wildcard inputs *).
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

    # Fallback that mirrors the original inline wrapper WHEN the module exists;
    # otherwise return the base model unchanged (quietly).
    def _apply_inline(self, model: ModelPatcher, method, tile_width, tile_height, tile_overlap, tile_batch_size):
        try:
            from tiled_diffusion import MixtureOfDiffusers, MultiDiffusion, SpotDiffusion
        except Exception:
            # tiled_diffusion not installed → keep base model unchanged
            return model

        compression = 4 if "CASCADE" in str(getattr(getattr(model, "model", None), "model_type", "")).upper() else 8
        tw_px, th_px, ov_px = _snap_sizes(tile_width, tile_height, tile_overlap, compression)

        if method == "MultiDiffusion":
            impl = MultiDiffusion()
        elif method == "SpotDiffusion":
            impl = SpotDiffusion()
        else:
            impl = MixtureOfDiffusers()

        impl.tile_width = tw_px // compression
        impl.tile_height = th_px // compression
        impl.tile_overlap = ov_px // compression
        impl.tile_batch_size = int(tile_batch_size)
        impl.compression = compression

        m = model.clone()
        m.set_model_unet_function_wrapper(impl)
        m.model_options["tiled_diffusion"] = True
        return m

    def apply(self, model, method, tile_width, tile_height, tile_overlap, tile_batch_size, grid_json=None):
        grid = _to_dict(grid_json)
        if isinstance(grid, dict):
            tw_g, th_g, ov_g, scale = _infer_from_grid(grid)
            s = max(1, int(scale))
            if tw_g is not None:  tile_width  = int(tw_g) * s
            if th_g is not None:  tile_height = int(th_g) * s
            if ov_g is not None:  tile_overlap = int(ov_g) * s

        # try official module
        patched = self._apply_via_td(model, method, tile_width, tile_height, tile_overlap, tile_batch_size)
        if patched is not None:
            return (patched,)

        # inline wrapper (if module exists) or base model unchanged
        return (self._apply_inline(model, method, tile_width, tile_height, tile_overlap, tile_batch_size),)


NODE_CLASS_MAPPINGS = {
    "EgregoraTiledDiffusionModel": EgregoraTiledDiffusionModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EgregoraTiledDiffusionModel": "Egregora Magick Diffusion (Model)",
}
