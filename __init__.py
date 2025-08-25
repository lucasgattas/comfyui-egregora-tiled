# __init__.py
# Package entrypoint for ComfyUI custom nodes.
# Exposes NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.
# Includes backward-compatibility aliases for older workflow keys.


# ---- Robust imports (relative or flat) ----
try:
from .egregora_core_tiling import EgregoraImageTileSplit
except Exception: # pragma: no cover
from egregora_core_tiling import EgregoraImageTileSplit


try:
from .egregora_regional_prompts import EgregoraTiledRegionalPrompt
except Exception: # pragma: no cover
from egregora_regional_prompts import EgregoraTiledRegionalPrompt


try:
from .egregora_mixture_of_diffusers import EgregoraMixtureOfDiffusers
except Exception: # pragma: no cover
from egregora_mixture_of_diffusers import EgregoraMixtureOfDiffusers




# ---- Required by ComfyUI ----
# Primary node keys used by new graphs
NODE_CLASS_MAPPINGS = {
"EgregoraImageTileSplit": EgregoraImageTileSplit,
"EgregoraTiledRegionalPrompt": EgregoraTiledRegionalPrompt,
"EgregoraMixtureOfDiffusers": EgregoraMixtureOfDiffusers,
# Backward-compatibility: keep older saved workflows loading
# (old name → new implementation)
"EgregoraTiledDiffusionModel": EgregoraMixtureOfDiffusers,
}


NODE_DISPLAY_NAME_MAPPINGS = {
"EgregoraImageTileSplit": "Image Tile Split (Egregora)",
"EgregoraTiledRegionalPrompt": "Tiled Regional Prompt (Egregora)",
# Display name matches the README short description
"EgregoraMixtureOfDiffusers": "Egregora Mixture of Diffusers (MoD)",
# Back-compat display alias
"EgregoraTiledDiffusionModel": "Egregora Mixture of Diffusers (MoD)",
}


# Optional metadata
__all__ = list(NODE_CLASS_MAPPINGS.keys())
__version__ = "0.3.0"
