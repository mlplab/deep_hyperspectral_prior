# coding: utf-8


import os
import h5py
import shutil
import scipy.io
import numpy as np


data_path = 'dataset/icvl'
h5_path = os.path.join(data_path, 'h5')
mat_path = os.path.join(data_path, 'mat')



def trans_h52mat(img_path):
    f = h5py.File(img_path, 'r')
    bands = np.array(f['bands'], dtype=np.float32)
    rad = np.array(f['rad'], dtype=np.float32).transpose(2, 1, 0)
    rgb = np.array(f['rgb'], dtype=np.float32).transpose(2, 1, 0)
    datas = {'bands': bands, 'data': rad, 'rgb': rgb}
    return datas


def make_icvl_data(img_dir, save_dir):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    img_list = os.listdir(img_dir)
    for name in img_list:
        img_name = name.split('/')[-1]
        datas = trans_h52mat(os.path.join(img_dir, name))
        scipy.io.savemat(os.path.join(save_dir, img_name), datas)

    return None


make_icvl_data(h5_path, mat_path)