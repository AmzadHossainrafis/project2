
"""
project: deep learning for infected cell segmentation

author Information
==================
name: amzad hossain rafi
email: amzad.rafi@northsouth.edu
github:

"""
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU, add, Conv2D, PReLU, ReLU, Concatenate, Activation, MaxPool2D, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow import keras

def basicblocks(input, filter, dilates = 1):
    x1 = Conv2D(filter, (3, 3), padding = 'same', dilation_rate = 1*dilates)(input)
    x1 = ReLU()(BatchNormalization()(x1))
    return x1

def RSU7(input, in_ch = 3, mid_ch = 12, out_ch = 3):
    hx = input
    #1
    hxin = basicblocks(hx, out_ch, 1)
    hx1 = basicblocks(hxin, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx1)
    #2
    hx2 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx3)
    #4
    hx4 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx4)
    #5
    hx5 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides = 2)(hx5)
    #6
    hx6 = basicblocks(hx, mid_ch, 1)
    #7
    hx7 = basicblocks(hx6, mid_ch, 2)

    #down
    #6
    hx6d = Concatenate(axis = -1)([hx7, hx6])
    hx6d = basicblocks(hx6d, mid_ch, 1)
    a,b,c,d = K.int_shape(hx5)
    hx6d=keras.layers.UpSampling2D(size=(2,2))(hx6d)

    #5
    hx5d = Concatenate(axis=-1)([hx6d, hx5])
    hx5d = basicblocks(hx5d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx4)
    hx5d = keras.layers.UpSampling2D(size=(2,2))(hx5d)

    # 4
    hx4d = Concatenate(axis=-1)([hx5d, hx4])
    hx4d = basicblocks(hx4d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx3)
    hx4d = keras.layers.UpSampling2D(size=(2,2))(hx4d)

    # 3
    hx3d = Concatenate(axis=-1)([hx4d, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    return output

def RSU6(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx1)
    #2
    hx2=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx3)
    #4
    hx4=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx4)
    #5
    hx5 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx5)
    #6
    hx6=basicblocks(hx,mid_ch,1)
    hx6=keras.layers.UpSampling2D((2, 2))(hx6)

    #5
    hx5d = Concatenate(axis=-1)([hx6, hx5])
    hx5d = basicblocks(hx5d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx4)
    hx5d = keras.layers.UpSampling2D(size=(2,2))(hx5d)

    # 4
    hx4d = Concatenate(axis=-1)([hx5d, hx4])
    hx4d = basicblocks(hx4d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx3)
    hx4d = keras.layers.UpSampling2D(size=(2,2))(hx4d)

    # 3
    hx3d = Concatenate(axis=-1)([hx4d, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    return output

def RSU5(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx1)
    #2
    hx2=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx3)
    #4
    hx4=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx4)
    #5
    hx5 = basicblocks(hx, mid_ch, 1)
    #hx5 = keras.layers.MaxPool2D((2, 2), strides=2)(hx5)
    hx5 = keras.layers.UpSampling2D((2, 2))(hx5)
    # 4
    hx4d = Concatenate(axis=-1)([hx5, hx4])
    hx4d = basicblocks(hx4d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx3)
    hx4d = keras.layers.UpSampling2D(size=(2,2))(hx4d)

    # 3
    hx3d = Concatenate(axis=-1)([hx4d, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    return output

def RSU4(input,in_ch=3,mid_ch=12,out_ch=3):
    hx=input
    #1
    hxin=basicblocks(hx,out_ch,1)
    hx1=basicblocks(hxin,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx1)
    #2
    hx2=basicblocks(hx,mid_ch,1)
    hx=keras.layers.MaxPool2D((2,2),strides=2)(hx2)
    #3
    hx3 = basicblocks(hx, mid_ch, 1)
    hx = keras.layers.MaxPool2D((2, 2), strides=2)(hx3)
    #4
    hx4=basicblocks(hx,mid_ch,1)
    hx4=keras.layers.UpSampling2D((2,2))(hx4)

    # 3
    hx3d = Concatenate(axis=-1)([hx4, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx2)
    hx3d = keras.layers.UpSampling2D(size=(2,2))(hx3d)

    # 2
    hx2d = Concatenate(axis=-1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 1)
    a, b, c, d = K.int_shape(hx1)
    hx2d = keras.layers.UpSampling2D(size=(2,2))(hx2d)

    # 1
    hx1d = Concatenate(axis=-1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output=keras.layers.add([hx1d,hxin])
    return output

def RSU4f(input, in_ch = 3, mid_ch = 12, out_ch = 3):
    hx=input
    #1
    hxin = basicblocks(hx, out_ch, 1)
    hx1 = basicblocks(hxin, mid_ch, 1)
    #2
    hx2=basicblocks(hx, mid_ch, 2)
    #3
    hx3 = basicblocks(hx, mid_ch, 4)
    #4
    hx4=basicblocks(hx, mid_ch, 8)

    # 3
    hx3d = Concatenate(axis = -1)([hx4, hx3])
    hx3d = basicblocks(hx3d, mid_ch, 4)

    # 2
    hx2d = Concatenate(axis = -1)([hx3d, hx2])
    hx2d = basicblocks(hx2d, mid_ch, 2)

    # 1
    hx1d = Concatenate(axis = -1)([hx2d, hx1])
    hx1d = basicblocks(hx1d, out_ch, 1)

    #output
    output = keras.layers.add([hx1d, hxin])
    return output
