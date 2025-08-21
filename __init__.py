# __init__.py
# Ensure ComfyUI imports our custom nodes on startup

# Safe imports (works whether loaded as package or loose .py files)
try:
    from .egregora_tile_split_merge import (
        EgregoraImageTileSplit,
        EgregoraLatentTileSplit,
        EgregoraVAEDecodeFromTiles,
    )
    from .egregora_tiled_regional_prompt import (
        EgregoraTiledRegionalPrompt,
    )
except ImportError:
    from egregora_tile_split_merge import (
        EgregoraImageTileSplit,
        EgregoraLatentTileSplit,
        EgregoraVAEDecodeFromTiles,
    )
    from egregora_tiled_regional_prompt import (
        EgregoraTiledRegionalPrompt,
    )

# Optional: metadata for ComfyUI Manager (not required, but helpful)
__all__ = [
    "EgregoraImageTileSplit",
    "EgregoraLatentTileSplit",
    "EgregoraVAEDecodeFromTiles",
    "EgregoraTiledRegionalPrompt",
]

__version__ = "0.1.1"
__author__ = "Your Name or Team"
__description__ = "Custom tiled split/merge and regional prompt nodes for ComfyUI"
