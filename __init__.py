# __init__.py
# Package entrypoint for ComfyUI custom nodes.
# Exposes NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.

# ---- Robust imports (relative or flat) ----
try:
    # New consolidated helpers
    from .egregora_core_tiling import (
        EgregoraImageTileSplit,
        EgregoraTiledRegionalPrompt,
    )
except Exception:
    from egregora_core_tiling import (
        EgregoraImageTileSplit,
        EgregoraTiledRegionalPrompt,
    )

try:
    # Model wrapper (renamed display name → "Egregora Magick Diffusion (Model)")
    from .egregora_tiled_model import EgregoraTiledDiffusionModel
except Exception:
    from egregora_tiled_model import EgregoraTiledDiffusionModel


# ---- Required by ComfyUI ----
NODE_CLASS_MAPPINGS = {
    "EgregoraImageTileSplit": EgregoraImageTileSplit,
    "EgregoraTiledRegionalPrompt": EgregoraTiledRegionalPrompt,
    "EgregoraTiledDiffusionModel": EgregoraTiledDiffusionModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EgregoraImageTileSplit": "Image Tile Split (Egregora)",
    "EgregoraTiledRegionalPrompt": "Tiled Regional Prompt (Egregora)",
    # Display name reflects the rename you wanted
    "EgregoraTiledDiffusionModel": "Egregora Magick Diffusion (Model)",
}

# Optional metadata
__all__ = list(NODE_CLASS_MAPPINGS.keys())
__version__ = "0.2.0"
