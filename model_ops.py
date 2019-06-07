from .model import unet
from .losses import dice_coef_loss
from .metrics import dice_coef, dice_coef_liver, dice_coef_tumor
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
    seed(_seed)
    set_seed(_seed)
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




