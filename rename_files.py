# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:45:04 2019

@author: Tom
"""

import os
from tqdm import tqdm
import shutil
src = 'data/test/images/'
dst = 'data/test/images2/'

for d in tqdm(os.listdir(src)):
    shutil.copyfile(src + d, dst + d[3:])