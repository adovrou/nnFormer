import json
import numpy as np
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import save_pickle

# this code is no tested

json_splits_path = "/splits_final.json"
output_pkl_path = "/splits_final.pkl"

with open(json_splits_path, 'r') as f:
    json_data = json.load(f)

splits = []
for fold_info in json_data:
    fold_dict = OrderedDict()
    fold_dict['train'] = np.array(fold_info['train'])
    fold_dict['val'] = np.array(fold_info['val'])
    splits.append(fold_dict)

# Save using nnFormer's save_pickle utility
save_pickle(splits, output_pkl_path)
