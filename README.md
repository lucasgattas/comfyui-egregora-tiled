# comfyui-egregora-tiled  

✨ **Tiled regional prompting + tiled VAE decode with seam-free blending for ComfyUI**  

---

## 🚀 Features  

### 🔹 Image/Latent Split  
- Flexible grid with overlap.  
- Supports spiral or row-major order.  

### 🔹 VAE Decode From Tiles  
- Decodes each latent tile and blends in pixel space using a distance-to-edge smoothstep window.  
- Disables taper on outer edges (**no black border**).  
- Normalizes weights exactly for clean seams.  

### 🔹 Tiled Regional Prompt  
- Per-tile positive/negative prompts.  
- Supports global text, blacklist words, and combine modes (prepend/append/replace).  

> ✅ Works with **SDXL (f=4)** and **SD1/2 (f=8)** automatically.  
> ✅ No external dependencies.  

---

## 📦 Installation  

### 1. Via ComfyUI Manager (recommended)  
1. Open **ComfyUI → Manager**.  
2. Select **Install from URL** and paste this repo URL.  
3. Click **Install** and then **Reload ComfyUI**.  

### 2. Manual Installation  
```bash
cd ComfyUI/custom_nodes
[git clone [https://github.com/lucasgattas/comfyui-egregora-tiled.git]
# restart ComfyUI
````

---

## 🧩 Nodes

### 🖼️ Image Tile Split (Egregora)

Splits an **IMAGE** `(B,H,W,C)` into tiles with overlap.
Returns a concatenated batch + `grid_json`.

### 🌀 Latent Tile Split (Egregora)

Splits a **LATENT** `{'samples': (B,C,H_l,W_l)}` according to `grid_json`.
Uses edge-replicate padding to keep all tiles equal and preserve border context.

### 🎨 VAE Decode From Tiles (Egregora)

Grid-guided tiled VAE decode with seam-free blending:

* Distance-to-edge smoothstep window per tile.
* No outer border taper.
* Exact normalization by accumulated weights.
* `feather=0` → uses overlap from split (recommended).
* Set `feather` manually to override.

### ✍️ Tiled Regional Prompt (Egregora)

Generates per-tile conditioning from multi-line or comma-separated lists.
Supports:

* `global_positive`
* `global_negative`
* `blacklist_words`
* `combine_mode`

---

## ⚡ Quick Start

1. Resize input to target size (e.g., `1024×1536`).
2. **Image Tile Split:** `tile_w=512`, `tile_h=512`, `overlap=128`.
3. **Tiled Regional Prompt →** connect to sampler with same `grid_json`.
4. **Latent Tile Split** (optional, depends on your graph).
5. **Sampler** (per-tile or with regional conditioning).
6. **VAE Decode From Tiles:** input latents\_list + grid\_json + VAE.
7. Preview your result.

---

## 💡 Tips (SDXL + ControlNet)

* Overlap **96–128 px** works well.
* Decode once via **VAE Decode From Tiles** (avoid per-tile decode).
* If seams persist, increase overlap or feather.

---

## ✅ Compatibility

* ComfyUI latest stable.
* Tested with **SDXL / SD1.5**.
* No extra pip installs.
