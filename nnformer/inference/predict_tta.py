import argparse
import os
from copy import deepcopy
from typing import Tuple, Union, List
import numpy as np
import torch
import SimpleITK as sitk
import shutil
import json
from math import pi
from multiprocessing import Pool

from monai.transforms import (
    RandRotated,
    RandZoomd,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
    Invertd,
    InvertibleTransform,
)
try:
    from monai.data import MetaTensor
except ImportError:
    MetaTensor = None

# Compatibility shim for batchgenerators >= 0.25 in modern Python environments
import pkgutil
import importlib
import batchgenerators.dataloading
import batchgenerators.transforms

for _pkg in [batchgenerators.dataloading, batchgenerators.transforms]:
    for _, _modname, _ in pkgutil.iter_modules(_pkg.__path__):
        try:
            _submod = importlib.import_module(f'{_pkg.__name__}.{_modname}')
            for _attr in dir(_submod):
                if not _attr.startswith('_') and not hasattr(_pkg, _attr):
                    setattr(_pkg, _attr, getattr(_submod, _attr))
        except Exception:
            pass

from nnformer.inference.predict import preprocess_multithreaded, check_input_folder_and_return_caseIDs
from nnformer.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnformer.postprocessing.connected_components import load_remove_save, load_postprocessing
from nnformer.training.model_restore import load_model_and_checkpoint_files
from nnformer.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from nnformer.paths import preprocessing_output_dir
from nnformer.utilities.task_name_id_conversion import convert_id_to_task_name
from batchgenerators.utilities.file_and_folder_operations import *
from nnformer.preprocessing.preprocessing import get_lowres_axis, get_do_separate_z, resample_data_or_seg
from nnformer.run.default_configuration import get_default_configuration


import inspect


def _safe_init(cls, **kwargs):
    """
    Safely instantiates a class by filtering kwargs to only those accepted by cls.__init__.
    Ensures backward compatibility across different MONAI versions (e.g. 'lazy' parameter in MONAI >= 1.2).
    """
    sig = inspect.signature(cls.__init__)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return cls(**kwargs)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**valid_kwargs)


def get_tta_transforms() -> list:
    """
    Defines the 10 TTA transforms applied to the input volume using MONAI:
    1. 'original': Identity / no transformation.
    2. RandRotated (range_x=5 deg, range_y=5 deg, range_z=10 deg)
    3. RandRotated (range_x=3 deg, range_y=3 deg, range_z=5 deg)
    4. RandZoomd (min_zoom=0.9, max_zoom=1.1)
    5. RandZoomd (min_zoom=0.95, max_zoom=1.05)
    6. RandFlipd (spatial_axis=2)
    7. RandGaussianNoised (std=0.01)
    8. RandGaussianNoised (std=0.03)
    9. RandAdjustContrastd (gamma=(0.9, 1.1))
    10. RandAdjustContrastd (gamma=(0.95, 1.05))
    """
    return [
        "original",
        _safe_init(RandRotated, keys=["img"], prob=1.0, range_x=5 * pi / 180, range_y=5 * pi / 180, range_z=10 * pi / 180,
                   mode="bilinear", padding_mode="border", lazy=False),
        _safe_init(RandRotated, keys=["img"], prob=1.0, range_x=3 * pi / 180, range_y=3 * pi / 180, range_z=5 * pi / 180,
                   mode="bilinear", padding_mode="border", lazy=False),
        _safe_init(RandZoomd, keys=["img"], prob=1.0, min_zoom=0.9, max_zoom=1.1,
                   mode="trilinear", padding_mode="edge", lazy=False),
        _safe_init(RandZoomd, keys=["img"], prob=1.0, min_zoom=0.95, max_zoom=1.05,
                   mode="trilinear", padding_mode="edge", lazy=False),
        _safe_init(RandFlipd, keys=["img"], prob=1.0, spatial_axis=2),
        _safe_init(RandGaussianNoised, keys=["img"], prob=1.0, std=0.01),
        _safe_init(RandGaussianNoised, keys=["img"], prob=1.0, std=0.03),
        _safe_init(RandAdjustContrastd, keys=["img"], prob=1.0, gamma=(0.9, 1.1)),
        _safe_init(RandAdjustContrastd, keys=["img"], prob=1.0, gamma=(0.95, 1.05))
    ]


def save_map_nifti(map_array: np.ndarray, out_fname: str, dct: dict, order: int = 1,
                   force_separate_z: bool = None, order_z: int = 0) -> None:
    """
    Resamples and saves a floating point uncertainty map (variance or entropy) back into raw patient coordinate
    space, restoring spacing, inverting crop bounding box, and applying original ITK orientation metadata.
    """
    current_shape = map_array.shape
    shape_original_after_cropping = dct.get('size_after_cropping')
    shape_original_before_cropping = dct.get('original_size_of_raw_data')

    if np.any(np.array(current_shape) != np.array(shape_original_after_cropping)):
        if force_separate_z is None:
            if get_do_separate_z(dct.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(dct.get('original_spacing'))
            elif get_do_separate_z(dct.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(dct.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(dct.get('original_spacing'))
            else:
                lowres_axis = None

        map_old_spacing = resample_data_or_seg(map_array[None], shape_original_after_cropping, is_seg=False,
                                                axis=lowres_axis, order=order, do_separate_z=do_separate_z, cval=0,
                                                order_z=order_z)[0]
    else:
        map_old_spacing = map_array

    bbox_orig = dct.get('crop_bbox')
    if bbox_orig is not None:
        bbox = deepcopy(bbox_orig)
        map_old_size = np.zeros(shape_original_before_cropping, dtype=np.float32)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + map_old_spacing.shape[c], shape_original_before_cropping[c]))
        map_old_size[bbox[0][0]:bbox[0][1],
                     bbox[1][0]:bbox[1][1],
                     bbox[2][0]:bbox[2][1]] = map_old_spacing
    else:
        map_old_size = map_old_spacing

    map_resized_itk = sitk.GetImageFromArray(map_old_size.astype(np.float32))
    map_resized_itk.SetSpacing(dct['itk_spacing'])
    map_resized_itk.SetOrigin(dct['itk_origin'])
    map_resized_itk.SetDirection(dct['itk_direction'])
    sitk.WriteImage(map_resized_itk, out_fname)


def compute_dice(pred1: np.ndarray, pred2: np.ndarray, num_classes: int) -> List[float]:
    """
    Computes Sørensen-Dice coefficient between two segmentation masks for each foreground class c in [1, num_classes - 1].
    Returns 1.0 if both masks are empty for a given class.
    """
    dice = []
    for c in range(1, num_classes):
        p1 = (pred1 == c)
        p2 = (pred2 == c)
        intersection = np.sum(p1 & p2)
        union = np.sum(p1) + np.sum(p2)
        if union == 0:
            dice.append(1.0)
        else:
            dice.append(float(2.0 * intersection / union))
    return dice


def predict_cases_tta(model: str, list_of_lists: List[List[str]], output_filenames: List[str], folds: Union[Tuple[int], List[int]],
                      save_npz: bool, num_threads_preprocessing: int, num_threads_nifti_save: int,
                      segs_from_prev_stage: List[str] = None, do_mirroring: bool = False, mixed_precision: bool = True,
                      overwrite_existing: bool = False, all_in_gpu: bool = False, step_size: float = 0.5,
                      checkpoint_name: str = "model_final_checkpoint", segmentation_export_kwargs: dict = None,
                      disable_uncertainty: bool = False, uncertainty_metrics: List[str] = None,
                      prob_output_folder: str = None) -> None:
    """
    Performs 10-pass Test-Time Augmentation (TTA) inference across given cases using MONAI transforms, inverts
    spatial augmentations, ensembles probabilities across passes and folds, calculates voxel-wise uncertainty maps
    (variance, Shannon entropy), computes pairwise DSC diversity, and asynchronously exports results.
    """
    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None:
        assert len(segs_from_prev_stage) == len(output_filenames)

    if uncertainty_metrics is None:
        uncertainty_metrics = ['variance', 'entropy']
    elif 'all' in uncertainty_metrics:
        uncertainty_metrics = ['variance', 'entropy']

    pool = Pool(num_threads_nifti_save)
    results = []

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if not isfile(j)]
        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]
        print("number of cases that still need to be predicted:", len(cleaned_output_files))

    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(model, folds, mixed_precision=mixed_precision, checkpoint_name=checkpoint_name)

    if segmentation_export_kwargs is None:
        if 'segmentation_export_params' in trainer.plans.keys():
            force_separate_z = trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0
    else:
        force_separate_z = segmentation_export_kwargs['force_separate_z']
        interpolation_order = segmentation_export_kwargs['interpolation_order']
        interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(trainer, list_of_lists, cleaned_output_files, num_threads_preprocessing,
                                             segs_from_prev_stage)
    
    transforms = get_tta_transforms()
    print(f"Loaded {len(transforms)} TTA passes (MONAI). Starting TTA prediction...")

    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data

        print(f"\nPredicting {output_filename} with 10-pass TTA...")
        all_pass_probs = []

        # Run each of the 10 TTA passes
        for pass_idx, t in enumerate(transforms):
            transform_name = "original" if t == "original" else t.__class__.__name__
            print(f"  [Pass {pass_idx + 1}/10] {transform_name}")

            # 1. Wrap preprocessed data into MONAI MetaTensor or standard Tensor
            vol_tensor = torch.from_numpy(d).float()
            if MetaTensor is not None:
                vol_input = MetaTensor(vol_tensor)
            else:
                vol_input = vol_tensor
            aug_data = {"img": vol_input} if t == "original" else t({"img": vol_input})
            aug_img = aug_data["img"]

            # Extract NumPy array for sliding window prediction in nnFormer
            aug_img_np = aug_img.detach().cpu().numpy() if isinstance(aug_img, torch.Tensor) else np.asarray(aug_img)

            # 2. Sliding window inference ensembled across all specified folds
            fold_probs = []
            for p in params:
                trainer.load_checkpoint_ram(p, False)
                trainer.network.eval()
                trainer.network.do_ds = False

                pad_kwargs = {'constant_values': 0}
                mirror_axes = trainer.data_aug_params['mirror_axes'] if do_mirroring else None

                res = trainer.network.predict_3D(
                    aug_img_np, do_mirroring=do_mirroring, mirror_axes=mirror_axes, use_sliding_window=True,
                    step_size=step_size, patch_size=trainer.patch_size, regions_class_order=trainer.regions_class_order,
                    use_gaussian=True, pad_border_mode='constant', pad_kwargs=pad_kwargs,
                    all_in_gpu=all_in_gpu, verbose=False, mixed_precision=mixed_precision
                )
                # res[1] has shape: (num_classes, X', Y', Z')
                fold_probs.append(res[1])

            pass_prob = np.mean(fold_probs, axis=0) if len(fold_probs) > 1 else fold_probs[0]

            # 3. Spatial inversion for invertible transforms
            if isinstance(t, InvertibleTransform):
                device = aug_img.device if isinstance(aug_img, torch.Tensor) else torch.device("cpu")
                pred_tensor = torch.from_numpy(pass_prob).to(device)

                if MetaTensor is not None:
                    aug_data["pred"] = MetaTensor(
                        pred_tensor,
                        meta=deepcopy(aug_img.meta) if hasattr(aug_img, "meta") else {},
                        applied_operations=deepcopy(aug_img.applied_operations) if hasattr(aug_img, "applied_operations") else []
                    )
                else:
                    aug_data["pred"] = pred_tensor
                    if "img_meta_dict" in aug_data:
                        aug_data["pred_meta_dict"] = deepcopy(aug_data["img_meta_dict"])
                    if "img_transforms" in aug_data:
                        aug_data["pred_transforms"] = deepcopy(aug_data["img_transforms"])

                inverter = _safe_init(
                    Invertd,
                    keys="pred",
                    transform=t,
                    orig_keys="img",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="img_meta_dict",
                    nearest_interp=False  # Preserve continuous probabilities
                )
                inverted_data = inverter(aug_data)
                aligned_prob = inverted_data["pred"]
                aligned_prob_np = aligned_prob.detach().cpu().numpy() if isinstance(aligned_prob, torch.Tensor) else np.asarray(aligned_prob)
            else:
                aligned_prob_np = pass_prob

            # Offload immediately to CPU float32 to prevent GPU memory bloat
            all_pass_probs.append(aligned_prob_np.astype(np.float32, copy=False))
            del aug_data, aug_img, fold_probs
            torch.cuda.empty_cache()

        # Stack across all 10 TTA passes: shape (10, num_classes, X, Y, Z)
        all_probs = np.stack(all_pass_probs, axis=0)
        num_total_passes = all_probs.shape[0]
        num_classes = trainer.num_classes

        # 4. Aggregation & Uncertainty Calculation
        # Mean class probabilities across passes
        mean_probs = np.mean(all_probs, axis=0)

        # Variance map: voxel-wise variance across passes, then mean across classes
        variance_map = np.mean(np.var(all_probs, axis=0), axis=0)

        # Shannon entropy map across classes
        entropy_map = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=0)

        # 5. Pairwise DSC Diversity calculation
        if hasattr(trainer, 'regions_class_order') and trainer.regions_class_order is not None:
            all_segs = []
            for p in all_probs:
                seg = np.zeros(p.shape[1:], dtype=np.int16)
                for i, c in enumerate(trainer.regions_class_order):
                    seg[p[i] > 0.5] = c
                all_segs.append(seg)
        else:
            all_segs = [p.argmax(axis=0) for p in all_probs]

        pairwise_dscs = []
        for i in range(num_total_passes):
            for j in range(i + 1, num_total_passes):
                dice = compute_dice(all_segs[i], all_segs[j], num_classes)
                pairwise_dscs.append(dice)

        pairwise_dscs = np.array(pairwise_dscs)  # shape: (45, num_classes - 1)
        mean_dscs_per_class = np.nanmean(pairwise_dscs, axis=0).tolist() if len(pairwise_dscs) > 0 else []
        mean_dsc_overall = float(np.nanmean(pairwise_dscs)) if len(pairwise_dscs) > 0 else 1.0

        dsc_json = {
            "pairwise_dscs": np.where(np.isnan(pairwise_dscs), None, pairwise_dscs).tolist(),
            "mean_dscs_per_class": mean_dscs_per_class,
            "mean_dsc_overall": mean_dsc_overall
        }

        dsc_fname = output_filename[:-7] + "_tta_dsc.json"
        print(f"Saving TTA pairwise DSC diversity metrics to: {dsc_fname}")
        with open(dsc_fname, 'w') as f:
            json.dump(dsc_json, f, indent=4)

        # 6. Transpose backward to raw coordinate axes if forward transposition was applied
        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            mean_probs = mean_probs.transpose([0] + [i + 1 for i in transpose_backward])
            variance_map = variance_map.transpose([i for i in transpose_backward])
            entropy_map = entropy_map.transpose([i for i in transpose_backward])

        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None

        # 7. Asynchronous export of uncertainty maps
        if not disable_uncertainty:
            if 'variance' in uncertainty_metrics:
                var_fname = output_filename[:-7] + "_uncertainty_variance.nii.gz"
                print(f"Exporting variance uncertainty map to: {var_fname}")
                results.append(pool.starmap_async(
                    save_map_nifti,
                    ((variance_map, var_fname, dct, interpolation_order, force_separate_z, interpolation_order_z),)
                ))

            if 'entropy' in uncertainty_metrics:
                ent_fname = output_filename[:-7] + "_uncertainty_entropy.nii.gz"
                print(f"Exporting entropy uncertainty map to: {ent_fname}")
                results.append(pool.starmap_async(
                    save_map_nifti,
                    ((entropy_map, ent_fname, dct, interpolation_order, force_separate_z, interpolation_order_z),)
                ))

        # 8. Asynchronous export of final ensembled segmentation mask
        bytes_per_voxel = 4
        if all_in_gpu:
            bytes_per_voxel = 2
        if np.prod(mean_probs.shape) > (2e9 / bytes_per_voxel * 0.85):
            print("Saving output temporarily to disk")
            np.save(output_filename[:-7] + ".npy", mean_probs)
            mean_probs = output_filename[:-7] + ".npy"

        if save_npz:
            npz_file = output_filename[:-7] + ".npz"
        else:
            npz_file = None

        if prob_output_folder is not None:
            prob_output_fname = join(prob_output_folder, os.path.basename(output_filename))
        else:
            prob_output_fname = None

        print(f"Exporting ensembled segmentation to: {output_filename}")
        results.append(pool.starmap_async(
            save_segmentation_nifti_from_softmax,
            ((mean_probs, output_filename, dct, interpolation_order, region_class_order,
              None, None,
              npz_file, None, force_separate_z, interpolation_order_z, True, prob_output_fname),)
        ))

    print("Inference finished. Waiting for segmentation and map exports to complete...")
    _ = [i.get() for i in results]

    # Postprocessing
    results = []
    pp_file = join(model, "postprocessing.json")
    if isfile(pp_file):
        print("Applying postprocessing from postprocessing.json...")
        shutil.copy(pp_file, os.path.abspath(os.path.dirname(output_filenames[0])))
        for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
        results.append(pool.starmap_async(load_remove_save,
                                          zip(output_filenames, output_filenames,
                                              [for_which_classes] * len(output_filenames),
                                              [min_valid_obj_size] * len(output_filenames))))
        _ = [i.get() for i in results]

    pool.close()
    pool.join()
    print("All predictions, uncertainty maps, and metrics saved successfully.")


def predict_from_folder_tta(model: str, input_folder: str, output_folder: str, folds: Union[Tuple[int], List[int]],
                            save_npz: bool, num_threads_preprocessing: int, num_threads_nifti_save: int,
                            lowres_segmentations: Union[str, None], part_id: int, num_parts: int,
                            do_mirroring: bool = False, mixed_precision: bool = True, overwrite_existing: bool = True,
                            mode: str = 'normal', overwrite_all_in_gpu: bool = None, step_size: float = 0.5,
                            checkpoint_name: str = "model_final_checkpoint", segmentation_export_kwargs: dict = None,
                            disable_uncertainty: bool = False, uncertainty_metrics: List[str] = None,
                            prob_output_folder: str = None) -> None:
    """
    Validates input folder, reads case IDs, configures job partitioning, and delegates to predict_cases_tta.
    """
    maybe_mkdir_p(output_folder)
    shutil.copy(join(model, 'plans.pkl'), output_folder)

    assert isfile(join(model, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file"
    expected_num_modalities = load_pickle(join(model, "plans.pkl"))['num_modalities']

    case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)

    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]

    if lowres_segmentations is not None:
        assert isdir(lowres_segmentations), "if lowres_segmentations is not None then it must point to a directory"
        lowres_segmentations = [join(lowres_segmentations, i + ".nii.gz") for i in case_ids]
        assert all([isfile(i) for i in lowres_segmentations]), "not all lowres_segmentations files are present."
        lowres_segmentations = lowres_segmentations[part_id::num_parts]
    else:
        lowres_segmentations = None

    if overwrite_all_in_gpu is None:
        all_in_gpu = False
    else:
        all_in_gpu = overwrite_all_in_gpu

    return predict_cases_tta(
        model=model,
        list_of_lists=list_of_lists[part_id::num_parts],
        output_filenames=output_files[part_id::num_parts],
        folds=folds,
        save_npz=save_npz,
        num_threads_preprocessing=num_threads_preprocessing,
        num_threads_nifti_save=num_threads_nifti_save,
        segs_from_prev_stage=lowres_segmentations,
        do_mirroring=do_mirroring,
        mixed_precision=mixed_precision,
        overwrite_existing=overwrite_existing,
        all_in_gpu=all_in_gpu,
        step_size=step_size,
        checkpoint_name=checkpoint_name,
        segmentation_export_kwargs=segmentation_export_kwargs,
        disable_uncertainty=disable_uncertainty,
        uncertainty_metrics=uncertainty_metrics,
        prob_output_folder=prob_output_folder
    )


def main():
    parser = argparse.ArgumentParser(description="Test-Time Augmentation (TTA) with Uncertainty Quantification for nnFormer")
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct order", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="Folder for saving predictions and uncertainty maps")
    parser.add_argument('-o_prob', "--prob_output_folder", required=False, default=None, help="Folder for saving probability maps")
    parser.add_argument('-t', '--task_name', help='Task name or task ID (required)', required=True)
    parser.add_argument('-tr', '--trainer_class_name', help='Name of the nnFormerTrainer used. Default: %s' % default_trainer, required=False, default=default_trainer)
    parser.add_argument('-ctr', '--cascade_trainer_class_name', help="Cascade trainer class name. Default: %s" % default_cascade_trainer, required=False, default=default_cascade_trainer)
    parser.add_argument('-m', '--model', help="2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres. Default: 3d_fullres", default="3d_fullres", required=False)
    parser.add_argument('-p', '--plans_identifier', help='Plans identifier. Default: %s' % default_plans_identifier, default=default_plans_identifier, required=False)
    parser.add_argument('-f', '--folds', nargs='+', default='None', help="Folds to use for prediction (e.g. 0, 0 1 2 3 4, or all)")
    parser.add_argument('-z', '--save_npz', required=False, action='store_true', help="Save softmax probs as npz")
    parser.add_argument('-l', '--lowres_segmentations', required=False, default='None')
    parser.add_argument("--part_id", type=int, required=False, default=0)
    parser.add_argument("--num_parts", type=int, required=False, default=1)
    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int)
    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int)
    parser.add_argument("--disable_mirroring", required=False, default=False, action="store_true",
                        help="Disable internal patch-level sliding window mirroring (recommended to keep False unless pure TTA is desired)")
    parser.add_argument("--disable_tta", required=False, default=False, action="store_true",
                        help="Alias for --disable_mirroring to match legacy predict_mc arguments")
    parser.add_argument("--overwrite_existing", required=False, default=False, action="store_true")
    parser.add_argument("--mode", type=str, default="normal", required=False)
    parser.add_argument("--all_in_gpu", type=str, default="None", required=False)
    parser.add_argument("--step_size", type=float, default=0.5, required=False)
    parser.add_argument('-chk', required=False, default='model_final_checkpoint', help="Checkpoint name (default: model_final_checkpoint)")
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False)

    # Uncertainty Quantification arguments
    parser.add_argument("-uncertainty_metrics", nargs='+', default=['variance', 'entropy'],
                        choices=['variance', 'entropy', 'all'],
                        help="Metrics to output: variance, entropy, or all. Default: variance entropy")
    parser.add_argument("-disable_uncertainty", action="store_true",
                        help="If set, uncertainty maps will not be computed or saved")

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    prob_output_folder = args.prob_output_folder
    part_id = args.part_id
    num_parts = args.num_parts
    folds = args.folds
    save_npz = args.save_npz
    lowres_segmentations = args.lowres_segmentations
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    do_mirroring = not (args.disable_mirroring or args.disable_tta)
    step_size = args.step_size
    overwrite_existing = args.overwrite_existing
    mode = args.mode
    all_in_gpu = args.all_in_gpu
    model = args.model
    trainer_class_name = args.trainer_class_name
    cascade_trainer_class_name = args.cascade_trainer_class_name
    task_name = args.task_name

    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)

    assert model in ["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"]

    if lowres_segmentations == "None":
        lowres_segmentations = None

    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None

    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    if model == "3d_cascade_fullres":
        trainer = cascade_trainer_class_name
    else:
        trainer = trainer_class_name

    model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" + args.plans_identifier)
    print("using model stored in ", model_folder_name)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    if not os.path.exists(join(model_folder_name, "plans.pkl")):
        plans_file = join(preprocessing_output_dir, task_name, args.plans_identifier + "_plans_3D.pkl")
        shutil.copy(plans_file, join(model_folder_name, "plans.pkl"))

    _ = get_default_configuration(model, task_name, trainer_class_name, args.plans_identifier)

    predict_from_folder_tta(
        model=model_folder_name,
        input_folder=input_folder,
        output_folder=output_folder,
        folds=folds,
        save_npz=save_npz,
        num_threads_preprocessing=num_threads_preprocessing,
        num_threads_nifti_save=num_threads_nifti_save,
        lowres_segmentations=lowres_segmentations,
        part_id=part_id,
        num_parts=num_parts,
        do_mirroring=do_mirroring,
        mixed_precision=not args.disable_mixed_precision,
        overwrite_existing=overwrite_existing,
        mode=mode,
        overwrite_all_in_gpu=all_in_gpu,
        step_size=step_size,
        checkpoint_name=args.chk,
        disable_uncertainty=args.disable_uncertainty,
        uncertainty_metrics=args.uncertainty_metrics,
        prob_output_folder=prob_output_folder
    )


if __name__ == "__main__":
    main()
