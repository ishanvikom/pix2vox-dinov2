# DINOv2-Enhanced Pix2Vox

Single-image 3D shape reconstruction by replacing the VGG16 encoder in [Pix2Vox](https://github.com/hzxie/Pix2Vox) with a frozen [DINOv2](https://github.com/facebookresearch/dinov2) ViT-B/14 backbone and a small trainable projection layer.

This is the code accompanying our 6.S058 final project report (Spring 2026).

**Authors:** Ishanvi Kommula, Tina Li (MIT)

## Result

On the standard 13-category ShapeNetCore single-view benchmark, swapping VGG16 for DINOv2 improves mean test IoU from **0.613 → 0.630** (+2.83% relative), with consistent gains on all 13 categories.

## What's in this repo

- `DINOv2_Pix2Vox_FINAL_v2.ipynb` — single Colab notebook containing the full pipeline: data extraction, encoder definition, model building, training, evaluation, and analysis.
- `requirements.txt` — Python dependencies.

The 3D modules (decoder, merger, refiner) are pulled directly from the upstream Pix2Vox repository at https://github.com/hzxie/Pix2Vox; the notebook clones that repo and uses it unmodified.

## How to reproduce

The notebook is designed for Google Colab with an A100 GPU. The same code runs on a local GPU with minor path edits.

### 1. Get the data

Download the ShapeNet renderings and voxel grids from the [3D-R2N2 dataset](http://3d-r2n2.stanford.edu/) (the standard 13-category subset). You should end up with two `.tgz` files:

- `ShapeNetRendering.tgz` (renderings)
- `ShapeNetVox32.tgz` (32³ voxel grids)

Place them in your Google Drive at `MyDrive/6S058_project/data/` (or update the paths in Section 2 of the notebook for a local setup).

### 2. Open the notebook in Colab

Upload `DINOv2_Pix2Vox_FINAL_v2.ipynb` to Colab and select an A100 GPU runtime.

### 3. Run the experiments

The notebook is split into sections. Section 7 contains the **only flag you change between runs**:

- `USE_DINOV2 = False` → baseline (VGG16 encoder)
- `USE_DINOV2 = True` → our system (DINOv2 encoder)

To reproduce the headline numbers, run the full notebook twice — once with each setting. Each 30-epoch run takes ~35 minutes on an A100. Checkpoints, training history, and per-category test results are saved to `MyDrive/6S058_project/checkpoints/{baseline,dinov2}/`.

### 4. Generate the figures

Section 14 reads both runs' test results and produces the per-category bar chart, the gain-vs-baseline scatter, and the training curves used in the report.

## Training configuration

- Backbone: frozen for both systems
- Optimizer: Adam, lr=1e-3 for projection/decoder/refiner, lr=1e-4 for merger
- LR schedule: ×0.5 at epoch 22
- Epochs: 30
- Batch size: 64
- Mixed precision (fp16) forward pass; BCE loss in fp32 for numerical stability
- Augmentation: random background, color jitter, Gaussian noise, horizontal flip, center crop

## Notes on the implementation

A few details that aren't obvious from the notebook structure:

- DINOv2's forward pass uses `get_intermediate_layers` to access the 256 final-block patch tokens; these are projected via `LayerNorm → Linear(768→256) → GELU` and reshaped/pooled to a 256×8×8 feature map matching what the Pix2Vox decoder expects.
- The encoder backbone is frozen via `requires_grad=False`, **not** `torch.no_grad()` — the latter can sever gradients to the projection layer in some autograd configurations.
- BCE loss is computed outside the autocast context to avoid fp16 underflow in the implicit log of the sigmoid output.

These are documented in Section 3.5 of the report.

## Acknowledgments

- Pix2Vox: Xie et al., "Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images" (ICCV 2019).
- DINOv2: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (TMLR 2024).
- ShapeNet: Chang et al., "ShapeNet: An Information-Rich 3D Model Repository" (2015).

## License

MIT License — see `LICENSE`.
