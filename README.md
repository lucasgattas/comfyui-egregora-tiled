# comfyui-egregora-tiled

**✨ Tiled Regional Prompting + Model-level Tiled Diffusion for ComfyUI**

---

## 🚀 Features

### 🔹 Image Tile Split

* Flexible grid with overlap (row‑major or spiral ordering).
* Emits a shared `grid_json` used by the other nodes so everything stays in sync.

### 🔹 Tiled Regional Prompt

* Per‑tile positive/negative prompts from lists or text.
* Global text (prepend/append/replace), blacklist filtering, and CLIP‑encode caching.
* Validates and snaps each region to the latent grid to avoid off‑by‑one artifacts.
* ✅ Accepts **Primitive → String (multiline)** directly (e.g., output pasted from **Florence‑2**), making it plug‑and‑play in upscaler workflows.

### 🔹 Egregora Mixture of Diffusers (Model)

* Wraps your model with tiled diffusion (Mixture/Multi/Spot Diffusion style).
* Reads the same `grid_json` and auto‑overrides tile sizes/overlap so your prompt regions and tiling match.
* Edge‑aware **Hann (sided) blending** removes faint outer borders while keeping seams invisible.
* SDXL‑aware crop micro‑conditioning (`add_time_ids`) per tile for stable composition.
* Graceful fallback if optional extras aren’t present.

> ✅ Works with **SDXL (f=4)** and **SD1/2 (f=8)** automatically via scale/latent alignment.
> ✅ No extra pip installs.

---

## 📦 Installation

### 1) Via ComfyUI Manager (recommended)

1. Open **ComfyUI → Manager**.
2. Select **Install from URL** and paste this repo URL:

```
https://github.com/lucasgattas/comfyui-egregora-tiled
```

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

* **tiles\_batch** – concatenated tiles
* **grid\_json** – grid metadata for downstream nodes

---

### ✍️ Tiled Regional Prompt (Egregora)

Builds per‑tile **CONDITIONING** (positive & negative) from lists or text.
Supports:

* `global_positive`, `global_negative`
* `blacklist_words`
* `combine_mode`: `prepend | append | replace`
* CLIP text‑encode cache for speed
* Bounds validation / snap‑to‑latent‑grid
* **Primitive → String (multiline)** input (ideal for Florence‑2 output)

**Inputs:** `clip`, `grid_json` (wildcard), `prompt_list` (wildcard)
**Outputs:** `positive`, `negative`

---

### 🧪 Egregora Mixture of Diffusers (Model)

Model patcher that enables tiled diffusion and keeps tile sizing in lockstep with your grid.

* Methods: **Mixture of Diffusers**, **MultiDiffusion**, **SpotDiffusion**
* Reads `grid_json` (wildcard) and overrides `tile_width`, `tile_height`, `tile_overlap` automatically
* **Edge‑aware sided Hann** weights (no halo on canvas borders)
* **Context‑expanded UNet** per tile (predict on a larger crop, then center‑crop back)
* Optional `tile_batch_size` to batch UNet calls when VRAM allows

**Input:** `model` (+ `grid_json`)
**Output:** `model` (tiled‑enabled)

---

## ⚡ Quick Start

1. **Resize** your input image to target resolution.
2. **Image Tile Split** → set `tile_w`, `tile_h`, `overlap` (e.g., `512/512/96–128`).
3. **Florence‑2 (or other captioner)** → produce per‑tile text → **Primitive → String (multiline)** → wire into **Tiled Regional Prompt** `prompt_list`.
4. **Tiled Regional Prompt** → connect the same `grid_json` and `clip`.
5. **Egregora Mixture of Diffusers (Model)** → connect the same `grid_json` so tiling matches your regions.
6. (Optional) **ControlNet Tile** to preserve original layout/consistency during upscale or edits.
7. **Sampler** → use the patched model and ERP conditioning.
8. **VAE Decode** → decode as usual.

**Upscaler example (high‑level):**

```
IMAGE ➜ Image Tile Split ➜ (per‑tile) Florence‑2 ➜ Primitive:String(multiline) ➜
Tiled Regional Prompt ➜ (optional) ControlNet Tile ➜ Egregora MoD (Model) ➜ KSampler ➜ VAE Decode
```

---

## 💡 Tips

* Bigger tiles + smaller overlap = faster; more overlap = smoother transitions. Tune per model.
* Reuse phrases across tiles to maximize CLIP encode caching.
* Increase `tile_batch_size` if VRAM allows to amortize UNet overhead.
* If you ever see seams, verify **splitter** and **MoD** share the same `grid_json`.

---

## ✅ Compatibility

* ComfyUI (latest stable)
* Tested with **SDXL** and **SD1.5**
* No additional Python packages required

---

## 🗺️ What Changed

* Streamlined to a 3‑node pipeline: **Image Tile Split → Tiled Regional Prompt → Egregora MoD (Model)**.
* Fixed border halos via **edge‑aware sided Hann** blending.
* SDXL crop micro‑conditioning per tile for robust global composition.
* Added **Primitive String (multiline)** handling for clean Florence‑2 → ERP connections.

---

## 📝 License

MIT

---

## 🙌 Credits

Inspired by Mixture‑of‑Diffusers / MultiDiffusion research and the broader ComfyUI community. Special thanks to everyone experimenting with regional prompting and tiled upscalers.
