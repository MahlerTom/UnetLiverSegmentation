# To create segmentation masks of liver and lesions for CT scans, we will use the U-net network, a state-of-the-art network aimed for segmentation tasks, with few variations.
#
# The U-net architecture, presented in \Cref{fig:u-net architecture}, contains two paths:
# 1. First path is the contraction path (also called as the encoder). It is used to capture the context in the image. The encoder is constructed from CNN and max-pooling layers.
# 2. The second path is the symmetric expanding path (also called as the decoder). It is used to enable precise localization using transposed convolutions.
#
# ![alt text](https://github.com/zhixuhao/unet/raw/master/img/u-net-architecture.png "U-net architecture")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam

def unet(pretrained_weights=None, shape=(256, 256, 1), filters=64, optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy']):
    # First 3 layers
    inputs = Input(shape=shape, name='input')    
    conv1 = Conv2D(filters=filters, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1_1')(inputs)   
    conv1 = Conv2D(filters=filters, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1_2')(conv1)
    
    # Second 3 layers
    pool1 = MaxPooling2D(pool_size=2, name='pool2_0')(conv1)
    conv2 = Conv2D(filters=filters*2, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2_1')(pool1)
    conv2 = Conv2D(filters=filters*2, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2_2')(conv2)
    
    # Third 3 layers
    pool2 = MaxPooling2D(pool_size=2, name='pool3_0')(conv2)
    conv3 = Conv2D(filters=filters*4, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3_1')(pool2)
    conv3 = Conv2D(filters=filters*4, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3_2')(conv3)
    
    # Fourth 3 layers
    pool3 = MaxPooling2D(pool_size=2, name='pool4_0')(conv3)
    conv4 = Conv2D(filters=filters*8, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4_1')(pool3)
    conv4 = Conv2D(filters=filters*8, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4_2')(conv4)
    
    # Fifth Dropout layers
    drop4 = Dropout(rate=0.5, name='drop5_0')(conv4)
    
    # Fifth 3 layers
    pool4 = MaxPooling2D(pool_size=2, name='pool5_0')(drop4)
    conv5 = Conv2D(filters=filters*16, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv5_1')(pool4)
    conv5 = Conv2D(filters=filters*16, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv5_2')(conv5)
    
    # sixth Dropout layers
    drop5 = Dropout(rate=0.5, name='drop6_0')(conv5)
    
    # seventh concat layer
    up6 = Conv2D(filters=filters*8, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv7_concat')(
        UpSampling2D(size=2, name='up7_concat')(drop5)
    )
    merge6 = concatenate([drop4, up6], axis=3, name='concat7_0')
        
    # Rest of seventh layer
    conv6 = Conv2D(filters=filters*8, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv7_1')(merge6)
    conv6 = Conv2D(filters=filters*8, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv7_2')(conv6)

    # Eighth concat layer
    up7 = Conv2D(filters=filters*4, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv8_concat')(
        UpSampling2D(size=2, name='up8_concat')(conv6)
    )
    merge7 = concatenate([conv3,up7], axis=3, name='concat8_0')
    
    # Rest of eighth layer
    conv7 = Conv2D(filters=filters*4, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv8_1')(merge7)
    conv7 = Conv2D(filters=filters*4, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv8_2')(conv7)

    # Ninth concat layer
    up8 = Conv2D(filters=filters*2, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv9_concat')(
        UpSampling2D(size=2, name='up9_concat')(conv7)
    )
    merge8 = concatenate([conv2,up8], axis=3, name='concat9_0')
    
    # Rest of Ninth layer
    conv8 = Conv2D(filters=filters*2, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv9_1')(merge8)
    conv8 = Conv2D(filters=filters*2, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv9_2')(conv8)

    # Tenth concat layer
    up9 = Conv2D(filters=filters, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv10_concat')(
        UpSampling2D(size=2, name='up10_concat')(conv8)
    )
    merge9 = concatenate([conv1,up9], axis=3, name='concat10_0')
    
    # Rest of Tenth layer
    conv9 = Conv2D(filters=filters, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv10_1')(merge9)
    conv9 = Conv2D(filters=filters, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv10_2')(conv9)
    conv9 = Conv2D(filters=2, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv10_3')(conv9)
    
    conv10 = Conv2D(filters=1, kernel_size=1, activation='sigmoid', name='conv11_0')(conv9)
    
    model = Model(inputs, conv10, name="unet")

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    #model.summary()
       
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model  