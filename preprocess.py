import tensorflow as tf

def preprocess_image(img_raw, norm=None, _resize=None):
    # Decode png image into grayscale (channels = 1)
    img_tensor = tf.image.decode_png(img_raw, channels=1)
    if _resize is not None:
        img_tensor = tf.image.resize(img_tensor, _resize) # [192, 192]
    if norm is not None:
        img_tensor /= norm # 255.0
    return img_tensor
  
def load_and_preprocess_image(img_path, norm=None, _resize=None):
    img_raw = tf.io.read_file(img_path)
    return preprocess_image(img_raw, norm=norm, _resize=_resize)

def load_and_preprocess_from_path_label(path, label, norm=None, _resize=None):  
    return load_and_preprocess_image(path, norm=norm, _resize=_resize), load_and_preprocess_image(label, norm=norm, _resize=_resize)