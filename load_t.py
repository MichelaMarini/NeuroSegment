#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:40:57 2019

@author: mmarini
"""

import glob,numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#---------------------------------------------------------

 #numero immagini
channels = 1 #blue channel
height = 512    #change with 48
width = 512    #change with 48
dataset_path_result = "./result/"


def get_datasets_test(Nimgs, image_dir):
    imgs = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(image_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print
            img = Image.open(image_dir+files[i])
            imgs[i] = np.asarray(img).astype(np.float32)

        return imgs
