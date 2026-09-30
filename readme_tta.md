# Test-Time Augmentation (TTA) with Uncertainty Quantification for nnFormer

## 1. Overview & Theoretical Background

This module provides a standalone, production-ready **Test-Time Augmentation (TTA)** and **Uncertainty Quantification (UQ)** inference pipeline for **nnFormer**.

In standard inference, a trained model predicts segmentation maps solely from the original test volume. In contrast, **Test-Time Augmentation (TTA)** perturbs the input volume across multiple passes using slight geometric and intensity variations, inverts spatial transformations back into the original patient coordinate system, and ensembles the aligned class probabilities.

This process yields two distinct advantages:

1. **Higher Segmentation Robustness & Accuracy**: Ensembling over stochastic perturbations acts as a regularizer, reducing boundary errors and sensitivity to scanner noise or subtle patient orientation differences.
2. **Voxel-wise Uncertainty Quantification**: Disagreements across passes reveal regions where the network is uncertain. This highlights ambiguous tumor boundaries, faint lesions, or out-of-distribution anatomical features.

---

## 2. Mathematical Formulations

Given a preprocessed case volume $X \in \mathbb{R}^{C \times D \times H \times W}$ and $T = 10$ TTA transformations:

### 2.1. Forward Transformation & Spatial Inversion

For each transformation $t \in \{1, \dots, T\}$:

1. $X_t = t(X)$
2. $P_t^{\text{raw}} = \text{Softmax}\left(\text{nnFormer}(X_t)\right)$
3. If $t$ involves spatial deformation (rotation, zoom, flip):
   $$P_t = t^{-1}\left(P_t^{\text{raw}}\right)$$
   using continuous (trilinear) interpolation (`nearest_interp=False`) to avoid premature discretization.
4. If $t$ is an intensity transformation (noise, contrast, identity):
   $$P_t = P_t^{\text{raw}}$$

All passes $\{P_1, \dots, P_T\}$ are aligned to the identical coordinate grid with shape $(T, K, D, H, W)$, where $K$ denotes the number of classes.

---

### 2.2. Ensembled Mean Class Probabilities

$$\bar{P}_{k}(x, y, z) = \frac{1}{T} \sum_{t=1}^{T} P_{t, k}(x, y, z)$$

The final segmentation mask $S(x, y, z)$ is generated via:
$$S(x, y, z) = \arg\max_{k} \bar{P}_{k}(x, y, z)$$
_(or via region-based thresholds if `regions_class_order` is defined in the model plans)._

---

### 2.3. Voxel-wise Variance Map

Quantifies prediction dispersion across the $T$ stochastic passes. We compute the variance across passes per class, then take the mean across all $K$ classes:

$$V(x, y, z) = \frac{1}{K} \sum_{k=1}^{K} \left( \frac{1}{T} \sum_{t=1}^{T} \left( P_{t, k}(x, y, z) - \bar{P}_{k}(x, y, z) \right)^2 \right)$$

---

### 2.4. Voxel-wise Shannon Entropy Map

Measures the information-theoretic entropy of the ensembled probability distribution across classes:

$$H(x, y, z) = -\sum_{k=1}^{K} \bar{P}_{k}(x, y, z) \ln \left( \bar{P}_{k}(x, y, z) + 10^{-8} \right)$$

---

### 2.5. Pairwise DSC Diversity Metric

Measures agreement between discrete segmentation masks produced by each TTA pass. For each pass $t$, we compute the discrete segmentation $S_t = \arg\max_k P_{t, k}$.

For every pair of passes $(i, j)$ with $1 \le i < j \le T$ and for each foreground class $k \in \{1, \dots, K-1\}$:

$$
\text{DSC}_{i, j, k} = \begin{cases}
\frac{2 \cdot |(S_i == k) \cap (S_j == k)|}{|S_i == k| + |S_j == k|}, & \text{if } |S_i == k| + |S_j == k| > 0 \\
1.0, & \text{if } |S_i == k| + |S_j == k| = 0
\end{cases}
$$

The resulting metric file summarizes pairwise scores, class-level means, and the overall mean DSC diversity.

---

## 3. The 10 MONAI Transformations

All transforms are implemented using `monai.transforms` and wrapped in `monai.data.MetaTensor` to track spatial operation history:

|  Pass  | Transform             | Type      | Parameters                                                                                              |     Invertible?     |
| :----: | :-------------------- | :-------- | :------------------------------------------------------------------------------------------------------ | :-----------------: |
| **1**  | `"original"`          | Identity  | None                                                                                                    |      Identity       |
| **2**  | `RandRotated`         | Spatial   | `range_x=5*pi/180`, `range_y=5*pi/180`, `range_z=10*pi/180`, `mode="bilinear"`, `padding_mode="border"` | **Yes** (`Invertd`) |
| **3**  | `RandRotated`         | Spatial   | `range_x=3*pi/180`, `range_y=3*pi/180`, `range_z=5*pi/180`, `mode="bilinear"`, `padding_mode="border"`  | **Yes** (`Invertd`) |
| **4**  | `RandZoomd`           | Spatial   | `min_zoom=0.9`, `max_zoom=1.1`, `mode="trilinear"`, `padding_mode="edge"`                               | **Yes** (`Invertd`) |
| **5**  | `RandZoomd`           | Spatial   | `min_zoom=0.95`, `max_zoom=1.05`, `mode="trilinear"`, `padding_mode="edge"`                             | **Yes** (`Invertd`) |
| **6**  | `RandFlipd`           | Spatial   | `spatial_axis=2`                                                                                        | **Yes** (`Invertd`) |
| **7**  | `RandGaussianNoised`  | Intensity | `std=0.01`                                                                                              |       Direct        |
| **8**  | `RandGaussianNoised`  | Intensity | `std=0.03`                                                                                              |       Direct        |
| **9**  | `RandAdjustContrastd` | Intensity | `gamma=(0.9, 1.1)`                                                                                      |       Direct        |
| **10** | `RandAdjustContrastd` | Intensity | `gamma=(0.95, 1.05)`                                                                                    |       Direct        |

---

## 4. CLI Usage & Arguments Reference

The script is invoked via `python -m nnformer.inference.predict_tta`.

### Basic Usage

```bash
CUDA_VISIBLE_DEVICES=1  python -m nnformer.inference.predict_tta \
    -i /path/to/imagesTs \
    -o /path/to/pred_nnFormer_test_TTA\
    -t Task001_LungNodule \
    -m 3d_fullres \
    -f 0 \
    -tr nnFormerTrainerV2_nnformer_lungNodule
```

### Complete CLI Argument Reference

| Argument                               | Description                                                               |          Default          |
| :------------------------------------- | :------------------------------------------------------------------------ | :-----------------------: |
| `-i`, `--input_folder`                 | Input folder containing raw test modalities (`<case_id>_0000.nii.gz`)     |        _Required_         |
| `-o`, `--output_folder`                | Output folder where segmentation masks and uncertainty maps will be saved |        _Required_         |
| `-t`, `--task_name`                    | Task name or ID (e.g., `Task001_LungNodule` or `1`)                       |        _Required_         |
| `-m`, `--model`                        | Model configuration (`3d_fullres`, `3d_lowres`, `3d_cascade_fullres`)     |       `3d_fullres`        |
| `-f`, `--folds`                        | Folds to use for prediction (e.g., `0`, `0 1 2 3 4`, or `all`)            |          `None`           |
| `-tr`, `--trainer_class_name`          | Name of the trainer class used for model training                         |     `default_trainer`     |
| `-ctr`, `--cascade_trainer_class_name` | Trainer class name for 3D cascade fullres stage                           | `default_cascade_trainer` |
| `-chk`, `--checkpoint_name`            | Checkpoint file name (without `.model` suffix)                            | `model_final_checkpoint`  |
| `-uncertainty_metrics`                 | Metrics to calculate and export: `variance`, `entropy`, or `all`          |    `variance entropy`     |
| `-disable_uncertainty`                 | Flag to skip uncertainty map calculation and saving                       |          `False`          |
| `-z`, `--save_npz`                     | Save ensembled softmax probability arrays as compressed `.npz` files      |          `False`          |
| `-o_prob`, `--prob_output_folder`      | Separate directory to save probability maps                               |          `None`           |
| `--step_size`                          | Step size for sliding-window inference patch overlap                      |           `0.5`           |
| `--disable_mirroring`                  | Disable sliding-window patch-level mirroring (runs pure MONAI TTA)        |          `False`          |
| `--disable_mixed_precision`            | Disable PyTorch Automatic Mixed Precision (AMP)                           |          `False`          |
| `--num_threads_preprocessing`          | Worker threads for multithreaded preprocessing                            |            `6`            |
| `--num_threads_nifti_save`             | Worker processes for background NIfTI export                              |            `2`            |
| `--part_id` / `--num_parts`            | Partition test cases across multiple GPUs or jobs                         |         `0` / `1`         |

---

## 5. Output Files & Formats

For each processed case `<case_id>`, the following files are produced in the output directory:

```
output_folder/
├── plans.pkl
├── <case_id>.nii.gz                         # Final ensembled segmentation mask
├── <case_id>_uncertainty_variance.nii.gz    # Voxel-wise variance uncertainty map
├── <case_id>_uncertainty_entropy.nii.gz     # Voxel-wise Shannon entropy uncertainty map
└── <case_id>_tta_dsc.json                   # Pairwise DSC diversity scores
```

### JSON Diversity Metrics Schema (`<case_id>_tta_dsc.json`)

```json
{
    "pairwise_dscs": [
        [0.9612, 0.9420],
        [0.9587, 0.9385],
        ...
    ],
    "mean_dscs_per_class": [
        0.9542,
        0.9318
    ],
    "mean_dsc_overall": 0.9430
}
```

- `pairwise_dscs`: List of 45 pairwise DSC arrays across the 10 TTA passes for each foreground class.
- `mean_dscs_per_class`: Average pairwise DSC for each foreground class across all 45 pairs.
- `mean_dsc_overall`: Grand mean DSC across all pairs and classes.

---

## 6. Visualizing Uncertainty Maps

All exported uncertainty maps are restored back to the original patient coordinate system (matching the raw CT/MRI volume's spacing, origin, direction matrix, and uncropped bounding box).

### 6.1. ITK-SNAP

1. Open the original image (`<case_id>_0000.nii.gz`) as the **Main Image**.
2. Go to **File -> Add Another Image** and select `<case_id>_uncertainty_variance.nii.gz` or `<case_id>_uncertainty_entropy.nii.gz`.
3. In the left panel, adjust the display color map (e.g. choose **Hot Metal** or **Jet**) and adjust the opacity slider.
4. Load the segmentation `<case_id>.nii.gz` via **Segmentation -> Open Segmentation**.
5. Inspect areas of high variance/entropy along the segmentation boundaries.

### 6.2. 3D Slicer

1. Drag and drop the original image, segmentation mask, and uncertainty maps into 3D Slicer.
2. In the **Volumes** module, select `<case_id>_uncertainty_entropy`.
3. Under **Display -> Lookup Table**, change the color map from `Grey` to `Heat` or `Cold to Hot`.
4. Use the slice viewer controls to overlay the uncertainty map over the anatomical background.

---

## 7. Performance & Optimization Tips

- **Sliding-Window Mirroring vs. MONAI TTA**: Standard nnFormer inference applies 8 mirror combinations per patch during sliding window inference (`do_mirroring=True`). When combined with 10 TTA passes, this equals $10 \times 8 = 80$ forward passes per patch. If inference speed is a priority, pass `--disable_mirroring` to run sliding window inference without internal patch reflection, reducing runtime by ~8x while preserving the diversity of the 10 MONAI augmentations.
- **VRAM Offloading**: The pipeline offloads aligned probability tensors to host CPU memory immediately after inversion to prevent Out-Of-Memory (OOM) errors during long multi-case predictions.
- **Multithreading**: NIfTI saving and coordinate resampling are performed asynchronously in background worker processes via Python `multiprocessing.Pool`, ensuring the GPU remains fully utilized.
