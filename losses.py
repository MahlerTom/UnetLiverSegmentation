# The model is defined with the *loss* of 1-Dice coef total. We use this loss definition because it enables the training process find hidden features of liver and lesions simultaneously. This loss decrements improves the segmentation task each epoch.
# *Note:* matrices scores are calculated with quantified predicted images.

from .metrics import dice_coef, dice_coef_tumor, dice_coef_liver
from .utils import replace_tensor

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dice_coef_tumor_loss(y_true, y_pred, smooth=1):
    y_true_liver = replace_tensor(y_true, 1.0, 0.0)
    return 1-dice_coef_tumor(y_true_liver, y_pred, smooth=smooth)

def dice_coef_liver_loss(y_true, y_pred, smooth=1):
    return 1-dice_coef_liver(y_true, y_pred) 
