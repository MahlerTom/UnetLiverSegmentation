# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:53:29 2019

@author: Tom
"""

import shutil
import os
from tqdm import tqdm

train_dst = 'data/flat_train/'
val_dst = 'data/flat_val/'

def flatten(src, dst, prefix='.png'):
    for d in tqdm(os.listdir(src)):
#        print("Copy " + d + "...")
        shutil.copyfile(src + d[:-4] + prefix, dst + d[:-4] + prefix)    
        
        
flatten(src='data/train/images/', dst=train_dst)