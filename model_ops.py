from .model import unet
from .losses import dice_coef_loss
from .metrics import dice_coef, dice_coef_liver, dice_coef_tumor
from .data import init_ds

import datetime
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from numpy.random import seed
from tensorflow.random import set_seed

def model_fit(
    table=None,
    train_images_paths=train_images_paths,
    train_masks_paths=train_masks_paths,
    val_images_paths=val_images_paths,
    val_masks_paths=val_masks_paths,
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