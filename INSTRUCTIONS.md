### Step 0

Installation and setup

```bash
export nnFormer_raw_data_base="your_path/DATASET/nnFormer_raw"
export nnFormer_preprocessed="your_path/DATASET/nnFormer_preprocessed"
export RESULTS_FOLDER="your_path/DATASET/nnFormer_trained_models"
```

### Step 1

Prepare dataset

nnUnet_raw \
&emsp; nnUnet_raw_data \
&emsp;&emsp; DatasetXXX_Name \
&emsp;&emsp;&emsp;&emsp; imagesTr \
&emsp;&emsp;&emsp;&emsp;&emsp; case_1_0000.nii.gz \
&emsp;&emsp;&emsp;&emsp;&emsp; case_2_0000.nii.gz \
&emsp;&emsp;&emsp;&emsp; imagesTs \
&emsp;&emsp;&emsp;&emsp; labelsTr \
&emsp;&emsp;&emsp;&emsp;&emsp; case_1.nii.gz \
&emsp;&emsp;&emsp;&emsp;&emsp; case_2.nii.gz \
&emsp; nnUnet_cropped_data \
nnUnet_preprocessed \
nnUnet_results\

- Create dataset.json
  Example: \
  { \
  &emsp; "name": "LungNodule", \
  &emsp; "modality": { \
  &emsp;&emsp; "0": "noNorm" # for already preprocessed data \
  &emsp; }, \
  &emsp; "labels": { \
  &emsp;&emsp; "background": 0, \
  &emsp;&emsp; "lung_nodule": 1 \
  &emsp; }, \
  &emsp; "numTraining": number_of_train_val_samples, \
  &emsp; "numTest": number_of_test_samples, \
  &emsp; "test": [ \
  &emsp;&emsp; "./imagesTs/data1.nii.gz", \
  &emsp;&emsp; "./imagesTs/data2.nii.gz" \
  &emsp;] \
  &emsp; "training": [ \
  &emsp;&emsp; { \
  &emsp;&emsp; "image": "./imagesTr/LIDC_0000.nii.gz", \
  &emsp;&emsp; "label": "./labelsTr/LIDC_0000.nii.gz" \
  &emsp;&emsp; }, \
  &emsp;&emsp; { \
  &emsp;&emsp; "image": "./imagesTr/LIDC_0001.nii.gz", \
  &emsp;&emsp; "label": "./labelsTr/LIDC_0001.nii.gz" \
  &emsp;&emsp; } \
  &emsp; ] \
  }

- Create splits_final.json \
  [ \
  &emsp; { \
  &emsp;&emsp; "train": [ \
  &emsp;&emsp;&emsp; case_1, \
  &emsp;&emsp;&emsp; case_2 \
  &emsp;&emsp; ], \
  &emsp;&emsp; "val": [ \
  &emsp;&emsp;&emsp; case_3, \
  &emsp;&emsp;&emsp; case_4 \
  &emsp;&emsp; ] \
  &emsp; } \
  ]

### Step 2

Plan and preprocess

> nnFormer_plan_and_preprocess -t DATASET_ID -pl2d None --verify_dataset_integrity

> nnFormer_plan_and_preprocess -t 1 -pl2d None --verify_dataset_integrity

### Step 3

Train model

> CUDA_VISIBLE_DEVICES=0 nnFormer_train 3d_fullres nnFormerTrainerV2 Task001_LungNodule 0

### Step 4

Run inference

> CUDA_VISIBLE_DEVICES=0 nnFormer_predict -i imagesTs -o inferTs/${name} -m 3d_fullres -t ${task} -f 0 -chk model_best -tr nnFormerTrainerV2
