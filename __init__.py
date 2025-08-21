# __init__.py
# Make sure ComfyUI imports our nodes on startup.

from .egregora_tile_split_merge import (
    EgregoraImageTileSplit,
    EgregoraLatentTileSplit,
    EgregoraVAEDecodeFromTiles,
)

# Optional: only needed if you want the prompt builder in the same repo
from .egregora_tiled_regional_prompt import (
    EgregoraTiledRegionalPrompt,
)

# Helpful metadata (not required by ComfyUI)
__all__ = [
    "EgregoraImageTileSplit",
    "EgregoraLatentTileSplit",
    "EgregoraVAEDecodeFromTiles",
    "EgregoraTiledRegionalPrompt",
]
__version__ = "0.1.0"
