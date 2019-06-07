from .preprocess import load_and_preprocess_image, load_and_preprocess_from_path_label

import random
import pathlib
import IPython.display as display
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_data(data_path, print_imgs=3, width=256, height=256):
    data_root = pathlib.Path(data_path)
    data_paths = list(data_root.glob('*'))
    data_paths = [str(path) for path in data_paths]
    data_paths.sort()
    print('Loaded', len(data_paths), 'image paths')  

    if print_imgs > 0:
        print('##########################################')
        print('Printing Example Images')
        print()

        for n in range(print_imgs):
            image_path = random.choice(data_paths)
            display.display(display.Image(image_path, width=width, height=height))
            print(image_path.split('/')[-1][:-4])
            
        print('##########################################')
    
    return data_paths

def image_name(path):
    return path.split('/')[-1][:-4]

def create_image_ds(image_paths, norm=None, _resize=None, AUTOTUNE=tf.data.experimental.AUTOTUNE):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    image_ds = path_ds.map(
        lambda image_path: load_and_preprocess_image(image_path, norm=norm, _resize=_resize), 
        num_parallel_calls=AUTOTUNE)
    return image_ds

def create_ds(image_paths, mask_paths, norm=None, _resize=None, AUTOTUNE=tf.data.experimental.AUTOTUNE):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    label_ds = ds.map(
        lambda image_path, mask_path: load_and_preprocess_from_path_label(image_path, mask_path, norm=norm, _resize=_resize), 
        num_parallel_calls=AUTOTUNE
    )
    return label_ds