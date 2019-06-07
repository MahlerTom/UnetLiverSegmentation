## Define metrices and loss function
# 
# We define the following **metrices scores** regarding our model, to evaluate the quality of the results:
# 
# *   Dice coef total - dice coefficient score for both liver and lesions segmentation.
# *   Dice coef liver - dice coefficient score for liver segmentation only.
# *   Presicion liver - Precision score for liver segmentation only.
# *   Recall liver - Recall score for liver segmentation only.
# *   Dice coef lesions - dice coefficient score for lesions segmentation only.
# *   Presicion lesions - Precision score for lesions segmentation only.
# *   Recall lesions - Recall score for lesions segmentation only.
# 
# *Note:* metrices scores are calculated with quantized predicted images.

from utils import prepare_liver, prepare_tumor

from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K
import numpy as np

def dice_coef_f(y_true_f, y_pred_f, smooth=1, K=K):
    """
    Dice = 2*TP/ (2TP + FP + FN)
        =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + smooth)

def dice_coef(y_true, y_pred, smooth=1, K=K):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return dice_coef_f(y_true_f, y_pred_f, smooth=smooth, K=K)

def dice_coef_np(y_true, y_pred, smooth=1):    
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    return dice_coef_f(y_true, y_pred, smooth=smooth, K=np)

def dice_coef_liver(y_true, y_pred, smooth=1):
  y_true_liver, y_pred_liver = prepare_liver(y_true, y_pred)
  return dice_coef(y_true_liver, y_pred_liver, smooth=smooth)
    
def dice_coef_tumor(y_true, y_pred, smooth=1):
  y_true_tumor, y_pred_tumor = prepare_tumor(y_true, y_pred)
  return dice_coef(y_true_tumor, y_pred_tumor, smooth=smooth)

def precision_liver(y_true, y_pred):
  y_true_liver, y_pred_liver = prepare_liver(y_true, y_pred)
  return Precision(y_true_liver*2, y_pred_liver*2)
    
def precision_tumor(y_true, y_pred):
  y_true_tumor, y_pred_tumor = prepare_tumor(y_true, y_pred)
  return Precision(y_true_tumor, y_pred_tumor)

def recall_liver(y_true, y_pred):
  y_true_liver, y_pred_liver = prepare_liver(y_true, y_pred)
  return Recall(y_true_liver*2, y_pred_liver*2)
    
def recall_tumor(y_true, y_pred):
  y_true_tumor, y_pred_tumor = prepare_tumor(y_true, y_pred)
  return Recall(y_true_tumor, y_pred_tumor)