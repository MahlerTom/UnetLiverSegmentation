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


## Basic dataset munipulations before training
# To train the model with this dataset we will decide:
# 
# *   If and how to shuffle the data.
# *   How to divide to batches.
# 
# Furthermore, we will want the training process:
# *   To be repeatable over the the dataset.
# *   To make the batches available for training as soon as possible.
# 
# These features can be easily added using the `tf.data` api.

def prepare_ds(ds, batch_size, buffer_size, shuffle=True):
    # Setting a shuffle buffer size as large as the dataset ensures that the data is completely shuffled.
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size=batch_size)

    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def init_ds(ds, images_paths, masks_paths, norm=255.0, _resize=[256, 256], batch_size=1, buffer_size=None, shuffle=True, verbose=0):
    if ds is None:
        if buffer_size is None:
            buffer_size = len(images_paths)
        if masks_paths is None:
            ds = create_image_ds(images_paths, norm=norm, _resize=_resize)
    else:
        ds = create_ds(images_paths, masks_paths, norm=norm, _resize=_resize)
    ds = prepare_ds(ds, batch_size=batch_size, buffer_size=buffer_size, shuffle=shuffle)
        
    if verbose > 0:
        print(ds)
    return ds