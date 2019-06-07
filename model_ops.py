from .model import unet
from .losses import dice_coef_loss
from .metrics import dice_coef, dice_coef_liver, dice_coef_tumor, dice_coef_np
from .data import init_ds, image_name
from .utils import quantizatize

import datetime
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.random import set_seed
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
from skimage.transform import resize
from sklearn.metrics import precision_score, recall_score
from prettytable import MSWORD_FRIENDLY

def model_fit(    
    train_images_paths,
    train_masks_paths,
    val_images_paths,
    val_masks_paths,
    table=None,
    _resize=[256, 256],
    norm=255.0,
    batch_size=32,
    filters=4,
    lr=1e-3,
    epochs=30,
    loss=dice_coef_loss,
    metrics=None,
    verbose=1, 
    pretrained_weights=None,
    train_ds=None,
    val_ds=None,
    callbacks=None,
    steps_per_epoch=None,
    validation_steps=None,
    prefix='',
    shuffle=True,
    patience=3,
    _seed=2,
):  
    start_time = time.time()
    input_shape = (_resize[0], _resize[1], 1)
    optimizer = Adam(lr=lr)

    run_time = datetime.datetime.now().strftime('%m%d-%H%M%S')
    run_name = f'{prefix}_{run_time}_lr{lr}_f{filters}_s{_resize[0]}_e{epochs}_b{batch_size}'
    log_dir = f'logs/fit/{run_name}'
  
    if steps_per_epoch is None:
        steps_per_epoch = len(train_images_paths) // batch_size
    if validation_steps is None:
        validation_steps = len(val_images_paths) // batch_size
    if metrics is None:
        metrics = [dice_coef, Precision(name='Percision'), Recall(name='Recall'), dice_coef_liver, dice_coef_tumor]
  
    # Print stat
    print(f'steps_per_epoch = {steps_per_epoch}, validation_steps = {validation_steps}')
    if table is None:
        from prettytable import PrettyTable
        table = PrettyTable(['Run', 'Name', 'Optimizer', 'Batch Size', 'Resize', 'Filters', 'Learning Rate', 'Epochs'])

    table.add_row([run_time, prefix, 'Adam', batch_size, _resize[0], filters, lr, epochs])
    print(table)

    # Create Datasets
    train_ds = init_ds(
        train_ds, 
        images_paths=train_images_paths, masks_paths=train_masks_paths, 
        norm=norm, _resize=_resize, batch_size=batch_size, shuffle=shuffle)
  
    val_ds = init_ds(
        val_ds, 
        images_paths=val_images_paths, masks_paths=val_masks_paths, 
        norm=norm, _resize=_resize, batch_size=batch_size, shuffle=shuffle)

    # Create Model
    model = unet(shape=input_shape, filters=filters, pretrained_weights=pretrained_weights, optimizer=optimizer, loss=loss, metrics=metrics)

    if pretrained_weights is None:
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        )
        early_stop = EarlyStopping(patience=patience, verbose=verbose)

        if callbacks is None:
            callbacks = []
        callbacks.append(tensorboard_callback)
        callbacks.append(early_stop)
        model.fit(
            train_ds,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=val_ds,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            shuffle=shuffle
        )  
        model.save_weights(f'{run_name}.h5')
    print(f'--- {(time.time() - start_time):.2f} seconds ---')
    return model, table

def model_predict(
    model,
    images_paths,
    masks_paths,
    ds=None,
    steps=None,     
    verbose=1,
    images_to_print=10,
    norm=255.0,
    _resize=[256, 256], # _resize=[128, 128]
    save_image_path=None,
):
    from skimage import io
    if steps is None:
        steps = len(images_paths)
    
    ds = init_ds(
        ds, 
        images_paths=images_paths, masks_paths=masks_paths, 
        norm=norm, _resize=_resize, batch_size=1, shuffle=False, verbose=verbose)
    
    pred = model.predict(ds, verbose=verbose, steps=steps)
    print(pred.shape)

    if images_to_print is None:
        images_to_print = len(images_paths)

    num_of_splt = 4
    splt = 1
    plt.figure(figsize=(num_of_splt*6, images_to_print*6))
    if save_image_path is not None:
        for i in range(len(images_paths)):
            Image.fromarray(quantizatize(np.squeeze(pred[i,:,:,:]), 3, 170.0)*255).convert('L').save(os.path.join(save_image_path, f'{image_name(images_paths[i])}.png'))
      
    for i in range(images_to_print):
        i_pred = np.squeeze(pred[i,:,:,:])

        plt.subplot(images_to_print, num_of_splt, splt)
        plt.imshow(i_pred, cmap='gray', vmin=0, vmax=1)
        plt.title(f'{i} pred {image_name(images_paths[i])}')
        plt.grid(False); plt.xticks([]); plt.yticks([])

        plt.subplot(images_to_print, num_of_splt, splt+1)
        plt.imshow(quantizatize(i_pred, 3, 170.0), cmap='gray', vmin=0, vmax=1)
        plt.title(f'{i} quant {image_name(images_paths[i])}')
        plt.grid(False); plt.xticks([]); plt.yticks([])

        if masks_paths is not None:
            plt.subplot(images_to_print, num_of_splt, splt+2)
            plt.imshow(io.imread(masks_paths[i], as_gray=True), cmap='gray', vmin=0, vmax=1)
            plt.title(f'{i} val {image_name(masks_paths[i])}')
            plt.grid(False); plt.xticks([]); plt.yticks([])    
            
        plt.subplot(images_to_print, num_of_splt, splt+3)
        plt.imshow(io.imread(images_paths[i], as_gray=True), cmap='gray', vmin=0, vmax=1)
        plt.title(f'{i} original {image_name(images_paths[i])}')
        plt.grid(False); plt.xticks([]); plt.yticks([])

        splt += num_of_splt
    
    plt.show()
    return pred

def model_evaluate(model, images_paths, masks_paths, ds=None, norm=255.0, _resize=[256,256], verbose=1): 
    ds = init_ds(
        ds, 
        images_paths=images_paths, masks_paths=masks_paths, 
        norm=norm, _resize=_resize, batch_size=1, shuffle=False)    
    return model.evaluate(ds, steps=len(images_paths), verbose=verbose)

def print_model_scores(
    pred,
    images_paths,
    masks_paths,
    images_to_print=10,
    model=None,
    ds=None,
    steps=None,    
    verbose=1,
    smooth=1,
    norm=255.0,
    _resize=[256, 256],
):    
    if pred is None:
        if steps is None:
            steps = len(images_paths),

        ds = init_ds(
            ds, 
            images_paths=images_paths, masks_paths=masks_paths, 
            norm=norm, _resize=_resize, batch_size=1, shuffle=False)

        pred = model.predict(ds, verbose=verbose, steps=steps)
        print(pred.shape)
    
    splt = 1
    num_of_splt = 6
    plt.figure(figsize=(num_of_splt*6, images_to_print*6))
    for i in range(images_to_print):
        y_pred = quantizatize(np.squeeze(pred[i,:,:,:]), 3, 170)
        y_true = io.imread(masks_paths[i], as_gray=True)
        y_true = resize(y_true, (256, 256))
        y_true = quantizatize(np.squeeze(y_true), 2, 254)

        y_pred_liver = y_pred.copy()
        y_pred_liver[y_pred_liver==1] = 0
        y_true_liver = y_true.copy()
        y_true_liver[y_true_liver==1] = 0

        y_pred_tumor = y_pred.copy()
        y_pred_tumor[y_pred_tumor==0.5] = 0
        y_true_tumor = y_true.copy()
        y_true_tumor[y_true_tumor==0.5] = 0

        plt.subplot(images_to_print, num_of_splt, splt)
        plt.imshow(y_pred, cmap='gray', vmin=0, vmax=1)
        plt.title(f'{i} y_pred {dice_coef_np(y_true, y_pred, smooth=smooth)}')
        plt.grid(False); plt.xticks([]); plt.yticks([])

        plt.subplot(images_to_print, num_of_splt, splt+1)
        plt.imshow(y_true, cmap='gray', vmin=0, vmax=1)
        plt.title(f'{i} y_true')
        plt.grid(False); plt.xticks([]); plt.yticks([])

        plt.subplot(images_to_print, num_of_splt, splt+2)
        plt.imshow(y_pred_liver, cmap='gray', vmin=0, vmax=1)
        plt.title(f'{i} y_pred_liver {dice_coef_np(y_true_liver, y_pred_liver, smooth=smooth)}')
        plt.grid(False); plt.xticks([]); plt.yticks([])

        plt.subplot(images_to_print, num_of_splt, splt+3)
        plt.imshow(y_true_liver, cmap='gray', vmin=0, vmax=1)
        plt.title(f'{i} y_true_liver')
        plt.grid(False); plt.xticks([]); plt.yticks([])

        plt.subplot(images_to_print, num_of_splt, splt+4)
        plt.imshow(y_pred_tumor, cmap='gray', vmin=0, vmax=1)
        plt.title(f'{i} y_pred_tumor {dice_coef_np(y_true_tumor, y_pred_tumor, smooth=smooth)}')
        plt.grid(False); plt.xticks([]); plt.yticks([])

        plt.subplot(images_to_print, num_of_splt, splt+5)
        plt.imshow(y_true_tumor, cmap='gray', vmin=0, vmax=1)
        plt.title(f'{i} y_true_tumor')
        plt.grid(False); plt.xticks([]); plt.yticks([])

        splt += num_of_splt
    
    plt.show()
    return pred

def print_model_score_table(
    score_table,
    pred,
    images_paths,
    masks_paths,
    rows_to_print=0,
    model=None,
    ds=None,
    steps=None,    
    norm=255.0,
    _resize=[256, 256], # _resize=[128, 128]
    verbose=1,
    smooth=1
):
    def format_score(score):
        return f'{score*100:.2f}'
    
    def print_if_valid(y_true, y_pred, _dice_coef_avg, _precision_avg, _recall_avg, _sum):
        _dice_coef = _precision = _recall = ''
        if len(np.unique(y_true)) > 1:
            _dice_coef_num = dice_coef_np(y_true, y_pred, smooth=smooth)
            _dice_coef = format_score(_dice_coef_num)
            _dice_coef_avg += _dice_coef_num

            _precision_num = precision_score(y_true, y_pred)
            _precision = format_score(_precision_num)
            _precision_avg += _precision_num

            _recall_num = recall_score(y_true, y_pred)
            _recall = format_score(_recall_num)        
            _recall_avg += _recall_num

            _sum += 1

        return _dice_coef, _precision, _recall, _dice_coef_avg, _precision_avg, _recall_avg, _sum  
  
  
    if rows_to_print == 0:
        rows_to_print = len(images_paths)
    if pred is None:
        if steps is None:
            steps = len(images_paths),

        ds = init_ds(
            ds, 
            images_paths=images_paths, masks_paths=masks_paths, 
            norm=norm, _resize=_resize, batch_size=1, shuffle=False)

        pred = model.predict(ds, verbose=verbose, steps=steps)
        print(pred.shape)
    
    t = PrettyTable(['ID', 'Image', 
                    'Dice All', 
    #                    'Percision All', 'Recall All',
                    'Dice Liver', 'Percision Liver', 'Recall Liver', 
                    'Dice Tumor', 'Percision Tumor', 'Recall Tumor',
                    ])
    t.set_style(MSWORD_FRIENDLY)

    _dice_coef_liver_avg = _precision_liver_avg = _recall_liver_avg = _liver_sum = 0
    _dice_coef_tumor_avg = _precision_tumor_avg = _recall_tumor_avg = _tumor_sum = 0
    for i in range(len(images_paths)):
        y_pred = np.squeeze(pred[i,:,:,:])
        y_pred_qf = quantizatize(y_pred, 3, 170).flatten()*2
        y_true = np.squeeze(resize(io.imread(masks_paths[i], as_gray=True), (256, 256)))
        y_true_qf = quantizatize(y_true, 2, 254).flatten()*2

        y_pred_qf_liver = y_pred_qf.copy()
        y_pred_qf_liver[y_pred_qf_liver==2] = 0
        y_true_qf_liver = y_true_qf.copy()
        y_true_qf_liver[y_true_qf_liver==2] = 0

        y_pred_qf_tumor = y_pred_qf.copy()
        y_pred_qf_tumor[y_pred_qf_tumor==1] = 0
        y_pred_qf_tumor[y_pred_qf_tumor==2] = 1
        y_true_qf_tumor = y_true_qf.copy()
        y_true_qf_tumor[y_true_qf_tumor==1] = 0
        y_true_qf_tumor[y_true_qf_tumor==2] = 1

            
        _dice_coef_liver, _precision_liver, _recall_liver, _dice_coef_liver_avg, _precision_liver_avg, _recall_liver_avg, _liver_sum = print_if_valid(y_true_qf_liver, y_pred_qf_liver, _dice_coef_liver_avg, _precision_liver_avg, _recall_liver_avg, _liver_sum)
        _dice_coef_tumor, _precision_tumor, _recall_tumor, _dice_coef_tumor_avg, _precision_tumor_avg, _recall_tumor_avg, _tumor_sum = print_if_valid(y_true_qf_tumor, y_pred_qf_tumor, _dice_coef_tumor_avg, _precision_tumor_avg, _recall_tumor_avg, _tumor_sum)

        if i < rows_to_print:
            t.add_row([
                i+1, image_name(images_paths[i]),
                format_score(dice_coef_np(y_true_qf, y_pred_qf, smooth=smooth)),
        #         precision_score(y_true_qf, y_pred_qf, average='weighted'), recall_score(y_true_qf, y_pred_qf, average='weighted'),
        #         precision_score(y_true_qf, y_pred_qf, average='weighted', sample_weight=[0, .5, .5]), recall_score(y_true_qf, y_pred_qf, average='weighted', sample_weight=[0, .5, .5]),
                _dice_coef_liver, _precision_liver, _recall_liver,
                _dice_coef_tumor, _precision_tumor, _recall_tumor
            ])
    print(t)
  
    with np.errstate(divide='ignore', invalid='ignore'):
        score_table.add_row([
            np.divide(_dice_coef_liver_avg, _liver_sum),
            np.divide(_precision_liver_avg, _liver_sum),
            np.divide(_recall_liver_avg, _liver_sum),

            np.divide(_dice_coef_tumor_avg, _tumor_sum),
            np.divide(_precision_tumor_avg, _tumor_sum),
            np.divide(_recall_tumor_avg, _tumor_sum)
        ])
    print(); print(); print(score_table)
    return pred

def model_fit_eval(
    score_table,
    eval_table,
    model_table,
    train_images_paths,
    train_masks_paths,
    val_images_paths,
    val_masks_paths,
    _resize=[256, 256],
    norm=255.0,
    batch_size=32,
    filters=4,
    lr=1e-3,
    epochs=50,
    loss=dice_coef_loss,
    metrics=None,
    verbose=1, 
    shuffle=True,
    patience=3,
    pretrained_weights=None,
    train_ds=None,
    val_ds=None,
    callbacks=None,
    steps_per_epoch=None,
    validation_steps=None,
    prefix='',
    pred_images_to_print=0,
    rows_to_print=None,
    smooth=0,
    print_model_scores_images=0,
    random_seed=2,
):
    seed(random_seed)
    set_seed(random_seed)
    model, _ = model_fit(table=model_table, train_images_paths=train_images_paths,
                    train_masks_paths=train_masks_paths, val_images_paths=val_images_paths,
                    val_masks_paths=val_masks_paths, _resize=_resize, norm=norm,
                    batch_size=batch_size, filters=filters, lr=lr, epochs=epochs,
                    loss=loss, metrics=metrics, verbose=verbose, shuffle=shuffle,
                    pretrained_weights=pretrained_weights, train_ds=train_ds,
                    val_ds=val_ds, callbacks=callbacks,
                    steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                    prefix=prefix, patience=patience
                    )

    eval_table.add_row(model_evaluate(model, norm=norm, _resize=_resize, verbose=verbose))
    print(eval_table)
    pred = None
    if rows_to_print is not None:
        pred = model_predict(model, images_to_print=pred_images_to_print)
        print_model_score_table(score_table=score_table, pred=pred, rows_to_print=rows_to_print, smooth=smooth)

        if print_model_scores_images > 0:
            print_model_scores(pred=pred, images_to_print=print_model_scores_images, smooth=smooth)

    return model, pred 