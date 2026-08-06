# Monte Carlo (MC) Dropout Inference in nnFormer

This repository supports running inference with Monte Carlo (MC) dropout to capture epistemic uncertainty using the `predict_mc.py` script.

## Usage

You can run the script similar to standard nnFormer inference, but with additional arguments specifically for configuring MC dropout.

```bash
python -m nnformer.inference.predict_mc -i <input_folder> -o <output_folder> -t <task_name_or_id> -m <model> [MC_ARGS]
```

### Example Command

Based on standard inference, to run MC predictions on test images:

```bash
CUDA_VISIBLE_DEVICES=0 python -m nnformer.inference.predict_mc -i imagesTs -o pred_nnFormer_test_mc -m 3d_fullres -t Task001_LungNodule -f 0 -mc_samples 10 -dropout_prob 0.1
```

### MC-Specific Arguments

- `-mc_samples`: The number of stochastic forward passes (Monte Carlo samples) to run for each input case. Default: `10`.
- `-dropout_prob`: The dropout probability set during inference to sample different network weights. Default: `0.1`.
- `-disable_uncertainty`: If provided, the script will skip calculating the uncertainty metrics (entropy, variance, DSC) and will only compute the mean segmentation mask.
- `-uncertainty_metrics`: (Optional) Specify which metrics to calculate. Default is `all`.

## Outputs

For each case, the script generates 4 files in the specified `output_folder`:

1. **Averaged predicted mask** (`<case_id>.nii.gz`): 
   - The final discrete segmentation, obtained by averaging the softmax probabilities across all MC samples and applying argmax.
2. **Entropy Map** (`<case_id>_entropy.nii.gz`): 
   - A voxel-wise uncertainty map calculated as the entropy of the averaged softmax probabilities: $H = - \sum \bar{p}_c \log(\bar{p}_c)$.
3. **Variance Map** (`<case_id>_variance.nii.gz`):
   - A voxel-wise uncertainty map calculated as the sum of variances of the softmax probabilities across all MC samples.
4. **Uncertainty JSON** (`<case_id>_uncertainty.json`):
   - A JSON file containing:
     - `pairwise_dscs`: Dice Similarity Coefficient for all combinations of the generated MC masks per class.
     - `mean_dsc_overall`: The global average DSC across all pairs and classes.
     - `mean_dsc_per_class`: The average DSC across all pairs for each class.
