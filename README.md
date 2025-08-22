# comfyui-egregora-tiled

✨ **Tiled regional prompting + model-level tiled diffusion for ComfyUI**

---

## 🚀 Features

### 🔹 Image Tile Split

* Flexible grid with overlap (row-major or spiral ordering).
* Emits a shared `grid_json` used by the other nodes so everything stays in sync.&#x20;

### 🔹 Tiled Regional Prompt

* Per-tile positive/negative prompts from lists or text.
* Global text (prepend/append/replace), blacklist filtering, and CLIP-encode caching.
* Validates and snaps each region to the latent grid to avoid off-by-one artifacts.&#x20;

### 🔹 Egregora Magick Diffusion (Model)

* Wraps your model with tiled diffusion (Mixture/Multi/Spot Diffusion).
* Accepts the same `grid_json` (wildcard input) and auto-overrides tile sizes/overlap so your prompt regions and tiling match.
* Falls back gracefully if the external tiled module isn’t present.&#x20;

> ✅ Works with **SDXL (f=4)** and **SD1/2 (f=8)** automatically (via scale/latent alignment).&#x20;
> ✅ No extra pip installs.

---

## 📦 Installation

### 1) Via ComfyUI Manager (recommended)

1. Open **ComfyUI → Manager**.
2. Select **Install from URL** and paste this repo URL.
3. Click **Install** and then **Reload ComfyUI**.

### 2) Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lucasgattas/comfyui-egregora-tiled.git
# restart ComfyUI
```

---

## 🧩 Nodes

### 🖼️ Image Tile Split (Egregora)

Splits an **IMAGE** `(B,H,W,C)` into overlapping tiles and returns:

* **tiles\_batch**: concatenated tiles
* **grid\_json**: grid metadata for downstream nodes&#x20;

---

### ✍️ Tiled Regional Prompt (Egregora)

Builds per-tile **CONDITIONING** (positive & negative) from lists or text.
Supports:

* `global_positive`, `global_negative`
* `blacklist_words`
* `combine_mode`: `prepend | append | replace`
* CLIP text-encode cache for speed
* Optional bounds validation/snap to latent grid&#x20;

**Inputs:** `clip`, `grid_json` (wildcard), `prompt_list` (wildcard)
**Outputs:** `positive`, `negative`

---

### 🧪 Egregora Magick Diffusion (Model)

Model patcher that enables tiled diffusion and keeps tile sizing in lockstep with your grid.

* Methods: **Mixture of Diffusers**, **MultiDiffusion**, **SpotDiffusion**
* Reads `grid_json` (wildcard) and overrides `tile_width`, `tile_height`, `tile_overlap` automatically
* `tile_batch_size` to batch UNet calls when VRAM allows&#x20;

**Input:** `model` (+ optional `grid_json`)
**Output:** `model` (tiled-enabled)

---

## ⚡ Quick Start

1. **Resize** your input to the target resolution.
2. **Image Tile Split** → set `tile_w`, `tile_h`, `overlap` (e.g., `512/512/96–128`).
3. **Tiled Regional Prompt** → connect the same `grid_json` and your `clip`; fill `prompt_list` (multi-line or comma-separated).
4. **Egregora Magick Diffusion (Model)** → connect the same `grid_json` so tile sizes/overlap match your prompt regions.
5. **Sampler** → use the patched model and the conditioning from the regional prompt.
6. **VAE Decode** → decode as usual (no special tiled decoder needed with this setup).&#x20;

---

## 💡 Tips

* Larger tiles + smaller overlap = faster; more overlap = smoother seams (tune for your model).
* Reuse identical phrases across tiles to benefit from CLIP encode caching.
* Increase `tile_batch_size` if VRAM allows to amortize UNet overhead.&#x20;

---

## ✅ Compatibility

* ComfyUI latest stable.
* Tested with **SDXL** and **SD1.5**.
* No additional Python packages required.&#x20;

---

## 🗺️ What Changed (compared to earlier versions)

* Removed legacy **Latent Tile Split** and **VAE Decode From Tiles** sections from the workflow.
* Added the renamed **Egregora Magick Diffusion (Model)** node and streamlined the 3-node pipeline:
  **Image Tile Split → Tiled Regional Prompt → Egregora Magick Diffusion (Model)**.
