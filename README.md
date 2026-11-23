# EMTrace

**EMTrace** places every tumour cell on the epithelioid→sarcomatoid
(epithelial–mesenchymal) continuum and measures how much of each core is
sarcomatoid. It is a new cell-graph algorithm for mesothelioma subtyping
(epithelioid / biphasic / sarcomatoid) built on foundation-model nucleus
features.

## What changed from MesoGraph

| | MesoGraph | EMTrace |
|---|---|---|
| Nucleus segmentation | QuPath | **CellViT++** (masks, cell types, cell tokens) |
| Deep features | ResNet, 512-d | **H-optimus-1** ViT-g/14, 1536-d, mask-pooled patch tokens per nucleus, plus tile context |
| Handcrafted features | QuPath measurements | ~250 from CellViT masks: shape, Fourier, Hu, H/E/OD intensity in nucleus + ring, 78 Haralick, cell types, neighbourhood |
| Normalisation | fitted on **all** cores (test leakage) | fitted per training fold; PCA on deep blocks |
| Graph | dense adjacency, unit-ambiguous radius | sparse radius graph in µm, edge features |
| Loss | pairwise ranking (margin 0.8 train / 1.0 eval) | vectorised ranking with one margin **+ WHO-proportion constraint** |
| Explanations | GNNExplainer | GNNExplainer **and** integrated gradients, per feature block, plus per-cell and per-edge maps |
| Evaluation | single run | repeats, group-aware splits, bootstrap CIs |

### Novel components (each toggleable in the config)

- **WHO-proportion constraint** (`loss.proportion`). Each cell gets a
  soft sarcomatoid assignment. The core's tumour-cell-weighted sarcomatoid
  fraction must satisfy E ≤ 10%, S ≥ 90%, and B with both components ≥ 10%.
  This mirrors the WHO definition of biphasic. A graph-smoothness term
  favours contiguous components, so the model outputs a **sarcomatoid %
  per core** (`sarcomatoid_pct`).
- **Epithelioid→sarcomatoid pseudotime** (`pseudotime`). A DDRTree-style
  principal tree is fitted to the learned tumour-cell embeddings of the
  training fold and rooted at the most epithelioid cluster. Each cell gets a
  position on the E→S continuum; each core gets a histogram, mean, median
  and bimodality coefficient of those positions.
- **Pathologist-concept features** (`extract.concepts`). CONCH scores a
  crop around each nucleus against prompts for spindle shape, pleomorphism,
  nucleoli, mitoses, necrosis, stroma type and more. These form a
  human-readable feature block.
- **Self-supervised graph pretraining** (`pretrain`, **on by default**).
  GraphMAE-style masked-feature reconstruction pretrains the cell encoder on
  the training fold's cores (and optional extra unlabelled cores) before
  supervised training.

## Install

```bash
pip install -e .                 # core
pip install -e ".[concepts]"     # + CONCH for concept features
```

H-optimus-1 (Bioptimus, CC BY-NC-ND 4.0) and CONCH (MahmoodLab) are gated on
Hugging Face. Accept their terms there and `export HF_TOKEN=...`. No model
weights are included in this repository.

## Data

One CSV row per core:

| column | meaning |
|---|---|
| `core_id` | unique id |
| `label` | `E`, `B` or `S` |
| `group` | slide or patient id, used for group-aware splits |
| `image` | RGB core image (png/jpg/tif) |
| `mpp` | microns per pixel of `image` |
| `cellvit_json` | CellViT++ cell output for this core |
| `cellvit_graph` | *(optional)* CellViT++ `--graph` .pt holding the cell embeddings |
| `offset_x`, `offset_y` | *(optional)* pixel offset if CellViT ran on the parent slide |

Run CellViT++ first (see its repo):
`python cellvit/detect_cells.py --model CellViT-SAM-H-x40-AMP.pth --outdir out --graph --geojson process_wsi --wsi_path core.tif`.
CellViT++ works at 0.25 µm/px; H-optimus-1 features are computed at 0.5 µm/px,
and resampling is handled automatically.

## Run

```bash
emtrace demo --work-dir runs/demo                 # synthetic data, no weights needed
emtrace extract --config configs/example.yaml     # features -> work_dir/features
emtrace graphs  --config configs/example.yaml     # graphs   -> work_dir/graphs
emtrace train   --config configs/example.yaml     # CV training (pretraining, proportion loss, pseudotime)
emtrace explain --config configs/example.yaml --fold rep0_fold0   # --method ig | gnnexplainer | ig,gnnexplainer
```

Outputs in `work_dir`:

- `core_predictions.csv`: score, sarcomatoid %, pseudotime summary
- `metrics.json`: AUCs and Kendall τ with bootstrap CIs
- per fold: model, normaliser, pseudotime tree, node predictions, training history
- after `explain`: `attributions_{ig,gnnexplainer}_{features,blocks}.csv`, and `explanations/<core>.npz` with per-cell IG, GNNExplainer cell importance and edge mask for overlays

## Tests

`pytest` runs unit tests and a full synthetic end-to-end pipeline (16 tests).

## Status and caveats

- Validated on synthetic data only. Real-data performance has not been
  benchmarked yet.
- The CellViT++ `--graph` .pt key names are not documented upstream. The
  reader tries common names and matches embeddings to cells by centroid.
  Check this on your first real core.
- If B labels are patient-level rather than core-level, a B-labelled core
  may contain only one component; lower `loss.proportion.weight`.

## Credits

Derived from MesoGraph by Eastwood et al. (CC BY-NC 4.0); see `NOTICE`.
CellViT++ (Hörst et al.), H-optimus-1 (Bioptimus), CONCH (Lu et al.) and
GraphMAE (Hou et al.) belong to their respective authors.

## License

All rights reserved; the code is public for viewing only. No permission is given to use, copy, modify or redistribute it without written permission. Portions derived from MesoGraph remain under CC BY-NC 4.0 (see `NOTICE`). See `LICENSE`.
