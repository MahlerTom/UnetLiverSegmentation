import tensorflow as tf

def quantizatize(img, q, scale, isfloat=True):
    q = 255 // q
    img_q = img*255 if isfloat else img
    img_q = (img_q // q) * q
    img_q = img_q / scale
    img_q = img_q if isfloat else img_q * 255
    return img_q

def replace_tensor(tensor, frm, to):
    return tf.where(tf.equal(frm, tensor), to*tf.ones_like(tensor), tensor)

def prepare_liver(y_true, y_pred):
    y_pred_liver = quantizatize(y_pred, 3, 170.0)
    y_pred_liver = replace_tensor(y_pred_liver, 1.0, 0.0)
    y_true_liver = replace_tensor(y_true, 1.0, 0.0)
    return y_true_liver, y_pred_liver

def prepare_tumor(y_true, y_pred):
    y_pred_tumor = quantizatize(y_pred, 3, 170.0)
    y_pred_tumor = replace_tensor(y_pred_tumor, 0.5, 0.0)
    y_true_tumor = replace_tensor(y_true, 0.5, 0.0)
    return y_true_tumor, y_pred_tumor