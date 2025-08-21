# comfyui-egregora-tiled
Tiled **regional prompting** + **tiled VAE decode** with seam-free blending.

- **Image/Latent Split:** flexible grid with overlap, spiral or row-major order.  
- **VAE Decode From Tiles:** decodes each latent tile, blends in pixel space using a distance-to-edge smoothstep window, and **disables taper on outer edges** (no black border), then **exactly normalizes** the weights.  
- **Tiled Regional Prompt:** build per-tile positive/negative prompts with optional global text, blacklisting, and combine modes.

> Works with SDXL (f=4) and SD1/2 (f=8) automatically.  
> No external dependencies.

## Install

### Via ComfyUI Manager (recommended)
1. Open **ComfyUI → Manager**.
2. **Install from URL** and paste this repo URL.
3. Click **Install** and then **Reload ComfyUI**.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<you>/comfyui-egregora-tiled.git
# restart ComfyUI
