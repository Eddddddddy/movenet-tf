from tqdm import tqdm
from PIL import Image
import h5py
import numpy as np
import io
import random
import json


def data_read2memory(cfg):
    dataset_h5 = h5py.File(cfg['dataset_h5'], 'r')
    datasetval_h5 = h5py.File(cfg['datasetval_h5'], 'r')

    interp_methods = [2, 3, 0, 1]

    with open(cfg['train_label_path'], 'r') as f:
        train_label_list = json.loads(f.readlines()[0])
        # random.shuffle(train_label_list)

    with open(cfg['val_label_path'], 'r') as f:
        val_label_list = json.loads(f.readlines()[0])

    input_data = [train_label_list, val_label_list]

    data_labels = input_data[0]

    dataset = {}

    for item in tqdm(data_labels):
        data = np.array(dataset_h5[item["img_name"]])  # write the data to hdf5 file
        img = Image.open(io.BytesIO(data))
        img = img.convert('RGB')
        img = img.resize((cfg['img_size'], cfg['img_size']), resample=random.choice(interp_methods))
        img = np.array(img)
        dataset[item["img_name"]] = img

    datasetval = {}
    data_labels = input_data[1]

    for item in tqdm(data_labels):
        data = np.array(datasetval_h5[item["img_name"]])  # write the data to hdf5 file
        img = Image.open(io.BytesIO(data))
        img = img.convert('RGB')
        img = img.resize((cfg['img_size'], cfg['img_size']), resample=random.choice(interp_methods))
        img = np.array(img)
        datasetval[item["img_name"]] = img

    print("data loaded")

    return dataset, datasetval
