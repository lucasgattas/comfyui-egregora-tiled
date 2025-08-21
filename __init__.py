# __init__.py
# Package entrypoint for ComfyUI custom nodes.
# Exposes NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.

# Import classes robustly (works whether imported as a package or loose files)
try:
    from .egregora_tile_split_merge import (
        EgregoraImageTileSplit,
        EgregoraLatentTileSplit,
        EgregoraVAEDecodeFromTiles,
    )
except Exception:
    from egregora_tile_split_merge import (
        EgregoraImageTileSplit,
        EgregoraLatentTileSplit,
        EgregoraVAEDecodeFromTiles,
    )

try:
    from .egregora_tiled_regional_prompt import (
        EgregoraTiledRegionalPrompt,
    )
except Exception:
    from egregora_tiled_regional_prompt import (
        EgregoraTiledRegionalPrompt,
    )

# ---- Required by ComfyUI ----
NODE_CLASS_MAPPINGS = {
    "EgregoraImageTileSplit": EgregoraImageTileSplit,
    "EgregoraLatentTileSplit": EgregoraLatentTileSplit,
    "EgregoraVAEDecodeFromTiles": EgregoraVAEDecodeFromTiles,
    "EgregoraTiledRegionalPrompt": EgregoraTiledRegionalPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EgregoraImageTileSplit": "Image Tile Split (Egregora)",
    "EgregoraLatentTileSplit": "Latent Tile Split (Egregora)",
    "EgregoraVAEDecodeFromTiles": "VAE Decode From Tiles (Egregora)",
    "EgregoraTiledRegionalPrompt": "Tiled Regional Prompt (Egregora)",
}

# Optional metadata
__all__ = list(NODE_CLASS_MAPPINGS.keys())
__version__ = "0.1.2"
