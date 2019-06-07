from .data import image_name, create_ds, prepare_ds

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import seed

def plot_images(dataset, n_images, images_paths=None, masks_paths=None):
    splt = 1
    num_of_splt = 2
    plt.figure(figsize=(num_of_splt*12, n_images*12))
    for i, (image, mask) in enumerate(dataset.take(n_images)):
        plt.subplot(n_images*2, num_of_splt, splt)
        plt.imshow(np.squeeze(image.numpy()), cmap='gray', vmin=0, vmax=1)
        if images_paths is not None:
            plt.title(f'{i} image {image_name(images_paths[i])}')
        plt.grid(False); plt.xticks([]); plt.yticks([])

        plt.subplot(n_images*2, num_of_splt, splt+1)
        plt.imshow(np.squeeze(mask.numpy()), cmap='gray', vmin=0, vmax=1)
        if masks_paths is not None:
            plt.title(f'{i} mask {image_name(masks_paths[i])}')
        plt.grid(False); plt.xticks([]); plt.yticks([])

        splt += num_of_splt          
    plt.show()

def flip(x, seed=None):
    """Flip augmentation
    Args:
        x: Image to flip
    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x, seed=seed)
    x = tf.image.random_flip_up_down(x, seed=seed)
    return x
  
def color(x, max_delta=0.05, seed=None):
    """Color augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """
    x = tf.image.random_brightness(x, max_delta=max_delta, seed=seed)
    return x
  
def rotate(x, seed=None):
    """Rotation augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32, seed=seed))
#   return tf.py_function(lambda image: io.transform.rotate(image, tf.random.uniform(shape=[], minval=-45, maxval=45, dtype=tf.int32, seed=seed), clip=True
#                 resize=True, mode='constant'
#                ), [x], Tout=tf.float32)

def zoom(x, seed=None):
    """Zoom augmentation
    Args:
        x: Image
    Returns:
        Augmented image
    """
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img, seed=None):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32, seed=seed)]

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32, seed=seed)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x, seed=seed))

def apply_da(ds, f):
    return ds.map(lambda x,y: (tf.clip_by_value(f(x), 0, 1), tf.clip_by_value(f(y), 0, 1)), num_parallel_calls=4)

def get_train_da(images_paths=train_images_paths, masks_paths=train_masks_paths, max_delta=0.1, norm=255.0, _resize=[256, 256], batch_size=32, shuffle=True):
  ds_original = create_ds(images_paths, masks_paths, norm=norm, _resize=_resize)
  ds_rotate = apply_da(create_ds(images_paths, masks_paths, norm=norm, _resize=_resize), lambda x: rotate(x))
  ds_color = apply_da(create_ds(images_paths, masks_paths, norm=norm, _resize=_resize), lambda x: color(x, max_delta=max_delta))
  
  ds_da = ds_original.concatenate(ds_rotate).concatenate(ds_color)
#   plot_images(ds_da, n_images=5)
  ds_da_size = 3*len(images_paths)
  ds_da = prepare_ds(ds_da, batch_size=batch_size, buffer_size=ds_da_size, shuffle=shuffle)
#   print(train_ds_da_size, train_ds_da)
  return ds_da, ds_da_size