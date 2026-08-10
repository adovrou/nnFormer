import argparse
import os
from copy import deepcopy
from typing import Tuple, Union, List
import numpy as np
import torch
import SimpleITK as sitk
import shutil
import json
from multiprocessing import Pool

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

def save_map_nifti(map_array, out_fname, dct, order=1, force_separate_z=None, order_z=0):
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

    bbox = dct.get('crop_bbox')
    if bbox is not None:
        map_old_size = np.zeros(shape_original_before_cropping)
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


def compute_dice(pred1, pred2, num_classes):
    dice = []
    for c in range(1, num_classes):
        p1 = (pred1 == c)
        p2 = (pred2 == c)
        intersection = np.sum(p1 & p2)
        union = np.sum(p1) + np.sum(p2)
        if union == 0:
            dice.append(np.nan)
        else:
            dice.append(2.0 * intersection / union)
    return dice


def predict_cases_mc(model, list_of_lists, output_filenames, folds, save_npz, num_threads_preprocessing,
                     num_threads_nifti_save, segs_from_prev_stage=None, do_tta=True, mixed_precision=True,
                     overwrite_existing=False, all_in_gpu=False, step_size=0.5, checkpoint_name="model_final_checkpoint",
                     segmentation_export_kwargs: dict = None, mc_samples=10, dropout_prob=0.1, disable_uncertainty=False,
                     uncertainty_metrics=None):
    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None: assert len(segs_from_prev_stage) == len(output_filenames)

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
    print("starting prediction...")
    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data

        print("predicting", output_filename)
        all_mc_softmax = []
        all_mc_segs = []

        for p in params:
            trainer.load_checkpoint_ram(p, False)
            
            trainer.network.eval()
            trainer.network.do_ds = False
            
            # Enable dropout and drop_path layers and set probability
            activated_count = 0
            for module in trainer.network.modules():
                classname = module.__class__.__name__
                if classname.startswith('Drop') or 'Dropout' in classname or 'DropPath' in classname:
                    module.train()
                    if hasattr(module, 'p'):
                        module.p = dropout_prob
                        activated_count += 1
                    if hasattr(module, 'drop_prob'):
                        module.drop_prob = dropout_prob
                        activated_count += 1
            print(f"Activated {activated_count} dropout/drop_path layers with probability {dropout_prob}")

            for i in range(mc_samples):
                # Ensure dropout remains in train mode before each sample
                for module in trainer.network.modules():
                    classname = module.__class__.__name__
                    if classname.startswith('Drop') or 'Dropout' in classname or 'DropPath' in classname:
                        module.train()

                pad_kwargs = {'constant_values': 0}
                mirror_axes = trainer.data_aug_params['mirror_axes'] if do_tta else None
                
                res = trainer.network.predict_3D(
                    d, do_mirroring=do_tta, mirror_axes=mirror_axes, use_sliding_window=True,
                    step_size=step_size, patch_size=trainer.patch_size, regions_class_order=trainer.regions_class_order,
                    use_gaussian=True, pad_border_mode='constant', pad_kwargs=pad_kwargs, 
                    all_in_gpu=all_in_gpu, verbose=True, mixed_precision=mixed_precision)
                
                all_mc_softmax.append(res[1][None])
                all_mc_segs.append(res[0])
                
        # Stack all softmaxes across MC samples and folds
        softmax = np.vstack(all_mc_softmax) 
        softmax_mean = np.mean(softmax, 0)
        
        # Transpose backward
        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax_mean = softmax_mean.transpose([0] + [i + 1 for i in transpose_backward])
            for i in range(len(all_mc_segs)):
                all_mc_segs[i] = all_mc_segs[i].transpose([i for i in transpose_backward])
                
        if hasattr(trainer, 'regions_class_order'):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None

        # Calculate metrics if not disabled
        if not disable_uncertainty:
            entropy_map = -np.sum(softmax_mean * np.log(softmax_mean + 1e-8), axis=0)
            # variance_map = np.sum(np.var(softmax, axis=0), axis=0)
            variance_map = np.mean(np.var(softmax, axis=0), axis=0)
            
            # Calculate pairwise DSC
            num_preds = len(all_mc_segs)
            num_classes = trainer.num_classes
            pairwise_dscs = []
            for i in range(num_preds):
                for j in range(i + 1, num_preds):  
                    # Quick debug print to confirm they are identical
                    # print(f"Are masks {i} and {j} exactly identical? {np.array_equal(all_mc_segs[i], all_mc_segs[j])}")

                    dice = compute_dice(all_mc_segs[i], all_mc_segs[j], num_classes)
                    pairwise_dscs.append(dice)
            
            pairwise_dscs = np.array(pairwise_dscs) # (pairs, classes-1)
            # handle nans
            mean_dsc_per_class = np.nanmean(pairwise_dscs, axis=0).tolist()
            mean_dsc_overall = np.nanmean(pairwise_dscs)
            
            unc_json = {
                "pairwise_dscs": np.where(np.isnan(pairwise_dscs), None, pairwise_dscs).tolist(),
                "mean_dsc_overall": float(mean_dsc_overall),
                "mean_dsc_per_class": mean_dsc_per_class
            }
            json_fname = output_filename[:-7] + "_uncertainty.json"
            with open(json_fname, 'w') as f:
                json.dump(unc_json, f, indent=4)
                
            # Save entropy and variance maps
            ent_fname = output_filename[:-7] + "_uncertainty_entropy.nii.gz"
            var_fname = output_filename[:-7] + "_uncertainty_variance.nii.gz"
            results.append(pool.starmap_async(save_map_nifti, ((entropy_map, ent_fname, dct, interpolation_order, force_separate_z, interpolation_order_z),)))
            results.append(pool.starmap_async(save_map_nifti, ((variance_map, var_fname, dct, interpolation_order, force_separate_z, interpolation_order_z),)))
            
        # Save mean segmentation
        bytes_per_voxel = 4
        if all_in_gpu:
            bytes_per_voxel = 2 
        if np.prod(softmax_mean.shape) > (2e9 / bytes_per_voxel * 0.85): 
            print("Saving output temporarily to disk")
            np.save(output_filename[:-7] + ".npy", softmax_mean)
            softmax_mean = output_filename[:-7] + ".npy"

        if save_npz:
            npz_file = output_filename[:-7] + ".npz"
        else:
            npz_file = None

        results.append(pool.starmap_async(save_segmentation_nifti_from_softmax,
                                          ((softmax_mean, output_filename, dct, interpolation_order, region_class_order,
                                            None, None,
                                            npz_file, None, force_separate_z, interpolation_order_z),)
                                          ))

    print("inference done. Now waiting for the segmentation export to finish...")
    _ = [i.get() for i in results]
    
    # now apply postprocessing
    results = []
    pp_file = join(model, "postprocessing.json")
    if isfile(pp_file):
        print("postprocessing...")
        shutil.copy(pp_file, os.path.abspath(os.path.dirname(output_filenames[0])))
        for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
        results.append(pool.starmap_async(load_remove_save,
                                          zip(output_filenames, output_filenames,
                                              [for_which_classes] * len(output_filenames),
                                              [min_valid_obj_size] * len(output_filenames))))
        _ = [i.get() for i in results]

    pool.close()
    pool.join()


def predict_from_folder_mc(model: str, input_folder: str, output_folder: str, folds: Union[Tuple[int], List[int]],
                           save_npz: bool, num_threads_preprocessing: int, num_threads_nifti_save: int,
                           lowres_segmentations: Union[str, None],
                           part_id: int, num_parts: int, tta: bool, mixed_precision: bool = True,
                           overwrite_existing: bool = True, mode: str = 'normal', overwrite_all_in_gpu: bool = None,
                           step_size: float = 0.5, checkpoint_name: str = "model_final_checkpoint",
                           segmentation_export_kwargs: dict = None, mc_samples: int = 10, dropout_prob: float = 0.2,
                           disable_uncertainty: bool = False, uncertainty_metrics: list = None):
    
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
        assert all([isfile(i) for i in lowres_segmentations]), "not all lowres_segmentations files are present. " 
        lowres_segmentations = lowres_segmentations[part_id::num_parts]
    else:
        lowres_segmentations = None

    if overwrite_all_in_gpu is None:
        all_in_gpu = False
    else:
        all_in_gpu = overwrite_all_in_gpu

    return predict_cases_mc(model, list_of_lists[part_id::num_parts], output_files[part_id::num_parts], folds,
                            save_npz, num_threads_preprocessing, num_threads_nifti_save, lowres_segmentations, tta,
                            mixed_precision=mixed_precision, overwrite_existing=overwrite_existing, all_in_gpu=all_in_gpu,
                            step_size=step_size, checkpoint_name=checkpoint_name,
                            segmentation_export_kwargs=segmentation_export_kwargs,
                            mc_samples=mc_samples, dropout_prob=dropout_prob, 
                            disable_uncertainty=disable_uncertainty, uncertainty_metrics=uncertainty_metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', help="Must contain all modalities for each patient in the correct order", required=True)
    parser.add_argument('-o', "--output_folder", required=True, help="folder for saving predictions")
    parser.add_argument('-t', '--task_name', help='task name or task ID, required.', default=default_plans_identifier, required=True)
    parser.add_argument('-tr', '--trainer_class_name', help='Name of the nnFormerTrainer used for 2D U-Net, full resolution 3D U-Net and low resolution U-Net. The default is %s.' % default_trainer, required=False, default=default_trainer)
    parser.add_argument('-ctr', '--cascade_trainer_class_name', help="Trainer class name used for predicting the 3D full resolution U-Net part of the cascade. Default is %s" % default_cascade_trainer, required=False, default=default_cascade_trainer)
    parser.add_argument('-m', '--model', help="2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres. Default: 3d_fullres", default="3d_fullres", required=False)
    parser.add_argument('-p', '--plans_identifier', help='do not touch this unless you know what you are doing', default=default_plans_identifier, required=False)
    parser.add_argument('-f', '--folds', nargs='+', default='None', help="folds to use for prediction")
    parser.add_argument('-z', '--save_npz', required=False, action='store_true', help="save softmax probs as npz")
    parser.add_argument('-l', '--lowres_segmentations', required=False, default='None')
    parser.add_argument("--part_id", type=int, required=False, default=0)
    parser.add_argument("--num_parts", type=int, required=False, default=1)
    parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int)
    parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int)
    parser.add_argument("--disable_tta", required=False, default=False, action="store_true")
    parser.add_argument("--overwrite_existing", required=False, default=False, action="store_true")
    parser.add_argument("--mode", type=str, default="normal", required=False)
    parser.add_argument("--all_in_gpu", type=str, default="None", required=False)
    parser.add_argument("--step_size", type=float, default=0.5, required=False)
    parser.add_argument('-chk', required=False, default='model_final_checkpoint')
    parser.add_argument('--disable_mixed_precision', default=False, action='store_true', required=False)
    
    # MC Dropout args
    parser.add_argument("-mc_samples", type=int, default=10, help="Number of MC samples for uncertainty estimation")
    parser.add_argument("-dropout_prob", type=float, default=0.1, help="Dropout probability during inference")
    parser.add_argument("-disable_uncertainty", action="store_true", help="If set, uncertainty metrics will not be computed")
    parser.add_argument("-uncertainty_metrics", nargs='+', default=['all'], help="Metrics to output. Default is all (entropy, variance, dsc)")

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    part_id = args.part_id
    num_parts = args.num_parts
    folds = args.folds
    save_npz = args.save_npz
    lowres_segmentations = args.lowres_segmentations
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    disable_tta = args.disable_tta
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
    
    _=get_default_configuration(model, task_name, trainer_class_name, args.plans_identifier)

    predict_from_folder_mc(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                           num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                           overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                           mixed_precision=not args.disable_mixed_precision, step_size=step_size, checkpoint_name=args.chk,
                           mc_samples=args.mc_samples, dropout_prob=args.dropout_prob,
                           disable_uncertainty=args.disable_uncertainty, uncertainty_metrics=args.uncertainty_metrics)

if __name__ == "__main__":
    main()
