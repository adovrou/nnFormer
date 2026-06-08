### Step 0

Installation and setup

```bash
export nnFormer_raw_data_base="your_path/DATASET/nnFormer_raw"
export nnFormer_preprocessed="your_path/DATASET/nnFormer_preprocessed"
export RESULTS_FOLDER="your_path/DATASET/nnFormer_trained_models"
```

### Step 1

Prepare dataset

```
nnFormer_raw/
└── nnFormer_raw_data
  └── DatasetXXX_Name
    └── imagesTr
      ├── case_1_0000.nii.gz
      ├── case_2_0000.nii.gz
    └── imagesTs
    └── labelsTr
      ├── case_1.nii.gz
      ├── case_2.nii.gz
└──  nnFormer_cropped_data
nnFormer_preprocessed/
nnFormer_results/
```

- Create dataset.json \
  Example:

```
  {
    "name": "LungNodule",
    "modality": {
      "0": "noNorm" # for already preprocessed data
    },
    "labels": {
      "background": 0,
      "lung_nodule": 1
    },
    "numTraining": number_of_train_val_samples,
    "numTest": number_of_test_samples,
    "test": [
      "./imagesTs/data1.nii.gz",
      "./imagesTs/data2.nii.gz"
    ],
    "training": [
       {
       "image": "./imagesTr/LIDC_0000.nii.gz",
       "label": "./labelsTr/LIDC_0000.nii.gz"
       },
       {
       "image": "./imagesTr/LIDC_0001.nii.gz",
       "label": "./labelsTr/LIDC_0001.nii.gz"
       }
    ]
  }

```

- Create splits_final.json

```
[
  {
    "train": [
      case_1,
      case_2
    ],
    "val": [
    case_3,
    case_4
    ]
  }
]
```

### Step 2

Plan and preprocess

```bash
nnFormer_plan_and_preprocess -t DATASET_ID -pl2d None --verify_dataset_integrity

nnFormer_plan_and_preprocess -t 1 -pl2d None --verify_dataset_integrity
```

### Step 3

Train model

> CUDA_VISIBLE_DEVICES=0 nnFormer_train 3d_fullres nnFormerTrainerV2 Task001_LungNodule 0


### Step 4

Run inference on test sets

```
CUDA_VISIBLE_DEVICES=0 nnFormer_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -m CONFIGURATION -t  DATASET_NAME_OR_ID -f 0

CUDA_VISIBLE_DEVICES=0 nnFormer_predict -i imagesTs -o pred_nnFormer_test -m 3d_fullres -t Task001_LungNodule -f 0

```
