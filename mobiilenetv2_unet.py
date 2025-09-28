import tensorflow as tf
from prompt_toolkit.input import Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

def conv_block(inputs, num_filters):
    x= Conv2D(num_filters,3, padding='same')(inputs)
    x= BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def  decoder_block(inputs, skip, num_filters):
    x= Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x= Concatenate()([x, skip])
    x= conv_block(x, num_filters)

    return x

def build_mobilenetv2_unet(input_shape):
    ''' Input '''
    inputs = Input(shape=input_shape)

    '''Pre-trained MobileNetV2'''
    encoder = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs, alpha=1.0)

    '''Encoder'''
    s1 = encoder.input
    s2 = encoder.get_layer("block_1_expand_relu").output
    s3 = encoder.get_layer("block_3_expand_relu").output
    s4 = encoder.get_layer("block_6_expand_relu").output

    '''Bridge'''
    b1 = encoder.get_layer("block_13_expand_relu").output

    '''Decoder'''
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(b1, s4, 256)
    d3 = decoder_block(b1, s4, 128)
    d4 = decoder_block(b1, s4, 64)

    '''Ouput'''
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    model = Model(inputs, outputs, name="MobileNetV2_U-Net")
    return model

if __name__ == '__main__':
    model = build_mobilenetv2_unet((512, 512, 3))
    model.summary()