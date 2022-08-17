# DnCNN Model
# ----------------------------------------------------------------------------------------------

"""
project: deep learning for infected cell segmentation

author Information
==================
name: amzad hossain rafi
email: amzad.rafi@northsouth.edu
github:

"""


from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, add, Conv2D, PReLU, ReLU, Concatenate, Activation, MaxPool2D, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda


def DnCNN(config):
    
    inpt = Input(shape=(config['height'], config['width'], config['in_channels']))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(config['num_classes'], (1, 1), activation='softmax',dtype='float32')(x)
    # x = Conv2D(filters=6, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    # x = tf.keras.layers.Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model
