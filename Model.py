# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
import math

from keras import backend as K, initializers
from keras.engine import Layer, InputSpec

K.set_image_dim_ordering('tf')

from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, Dropout
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization, Permute, Conv2D, AveragePooling2D, concatenate, activations, Bidirectional, GRU, Flatten, LSTM, add, Multiply, Activation
from keras.layers import Conv2D, MaxPooling2D, Add, LSTM, Flatten, Conv1D, MaxPooling1D, Dense, Activation, \
    Dropout, GlobalMaxPooling1D, AveragePooling2D, ConvLSTM2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Recurrent, Reshape, Bidirectional, \
    BatchNormalization, Merge, concatenate, activations, merge, add, Multiply
from keras.callbacks import TensorBoard
from keras.initializers import Initializer
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from data import image_size_dict
import os
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def get_model(img_rows, img_cols, num_PC, nb_classes, dataID=1, type='AMFRS', lr=0.001):
    if num_PC == 0:
        num_PC = image_size_dict[str(dataID)][2]
    if type == 'AMFRS':
        model = AMFRS(img_rows, img_cols, num_PC, nb_classes)
    else:
        print('invalid model type, default use AMFRS model')
        model = AMFRS(img_rows, img_cols, num_PC, nb_classes)

    rmsp = RMSprop(lr=lr, rho=0.9, epsilon=1e-05)
    model.compile(optimizer=rmsp, loss='categorical_crossentropy',
                          metrics=['accuracy'])
    return model

class Symmetry(Initializer):
    """N*N*C Symmetry initial
    """
    def __init__(self, n=200, c=16, seed=0):
        self.n = n
        self.c = c
        self.seed = seed

    def __call__(self, shape, dtype=None):
        rv = K.truncated_normal([self.n, self.n, self.c], 0., 1e-5, dtype=dtype, seed=self.seed)
        rv = (rv + K.permute_dimensions(rv, pattern=(1, 0, 2))) / 2.0
        return K.reshape(rv, [self.n * self.n, self.c])



def stdpooling(x):
    std = K.std(x, axis=1, keepdims=False)
    return std

def get_callbacks(decay=0.0001):
    def step_decay(epoch, lr):
        return lr * math.exp(-1 * epoch * decay)

    callbacks = []
    callbacks.append(LearningRateScheduler(step_decay))

    return callbacks



def attention_horizontal_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim1 = int(inputs.shape[1])
    input_dim2 = int(inputs.shape[2])
    input_dim3 = int(inputs.shape[3])

    a = Permute((3, 1,2))(inputs)
    a = Reshape((input_dim3, input_dim2,input_dim1))(a)
    a = Dense(input_dim2, activation='softmax',trainable=False)(a)

    a_probs = Permute((2,3,1))(a)
    # output_attention_mul = merge([inputs, a_probs], mode='mul')
    return a_probs

def attention_vertical_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim1 = int(inputs.shape[1])
    input_dim2 = int(inputs.shape[2])
    input_dim3 = int(inputs.shape[3])

    a = Permute((3, 2,1))(inputs)
    a = Reshape((input_dim3, input_dim2,input_dim1))(a)
    a = Dense(input_dim2, activation='softmax',trainable=False)(a)

    a_probs = Permute((3,2,1))(a)
    # output_attention_mul = merge([inputs, a_probs], mode='mul')
    return a_probs



class AMG(Layer):
    def __init__(self, mode='invert', **kwargs):
        super(AMG, self).__init__(**kwargs)
        self.mode = mode

    def call(self, x):
        if self.mode == 'background_enhance':

            mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            std = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axis=[1, 2], keepdims=True))

            mask = tf.sigmoid(std)

            mask = tf.tile(mask, [1, tf.shape(x)[1], tf.shape(x)[2], 1])  # 扩展到 (batch_size, height, width, channels)

            F_it = x * (1 - mask) + (1 - x) * mask
            return [F_it, mask]
        else:
            return x

    def compute_output_shape(self, input_shape):
        if self.mode == 'background_enhance':
            F_it_shape = input_shape
            mask_shape = input_shape
            return [F_it_shape, mask_shape]
        else:
            return input_shape





class FR(Layer):

    def __init__(self, activation=None, **kwargs):
        super(FR, self).__init__(**kwargs)

    def call(self, x):
        assert isinstance(x, list)
        F, A1, A2, mask = x

        mask_resized = tf.image.resize_images(
            mask,
            size=[tf.shape(A1)[1], tf.shape(A1)[2]],
            method=tf.image.ResizeMethod.BILINEAR
        )
        mask_proj = Conv2D(1, 15, padding='same')(mask_resized)
        mask_proj = tf.tile(mask_proj, [1, 1, 1, tf.shape(A1)[-1]])


        print("A1 shape:", A1.shape)
        print("mask_resized shape:", mask_resized.shape)
        print("mask_proj shape:", mask_proj.shape)

        one = tf.ones_like(A2)
        zero = tf.zeros_like(A2)
        AA2 = tf.where(((A1 + A2) / 2 < mask_proj), zero, one)


        print("AA2 shape:", AA2.shape)

        AA2_transposed = tf.transpose(AA2, perm=[3, 1, 2, 0])
        window_size = 2

        H = tf.shape(AA2_transposed)[1]
        W = tf.shape(AA2_transposed)[2]
        B = tf.shape(AA2_transposed)[3]
        C = tf.shape(AA2_transposed)[0]

        pad_H = (window_size - (H % window_size)) % window_size
        pad_W = (window_size - (W % window_size)) % window_size

        AA2_padded = tf.pad(AA2_transposed, [[0,0], [0,pad_H], [0,pad_W], [0,0]])

        merged = tf.reshape(AA2_padded, [C * B, tf.shape(AA2_padded)[1], tf.shape(AA2_padded)[2], 1])

        pooled = tf.nn.max_pool(
            merged,
            ksize=[1, window_size, window_size, 1],
            strides=[1, window_size, window_size, 1],
            padding='VALID'
        )

        upsampled = tf.image.resize_nearest_neighbor(
            pooled,
            [tf.shape(AA2_padded)[1], tf.shape(AA2_padded)[2]]
        )

        upsampled = tf.reshape(upsampled, [C, B, tf.shape(AA2_padded)[1], tf.shape(AA2_padded)[2]])
        upsampled = tf.transpose(upsampled, perm=[0, 2, 3, 1])  # [C, H_pad, W_pad, B]

        upsampled = upsampled[:, :H, :W, :]

        upsampled = tf.transpose(upsampled, perm=[3, 1, 2, 0])

        filled_mask = upsampled
        F1 = tf.multiply(F, filled_mask)
        F2 = tf.multiply(F, AA2)

        return [F1, F2]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        base_shape = input_shape[0]
        return [base_shape, base_shape]







class FS(Layer):

    def __init__(self, Thr3=0.5, activation=None, **kwargs):
        self.Thr3 = Thr3
        # self.activation = activations.get(activation)
        super(FS, self).__init__(**kwargs)


    def call(self, x):
        assert isinstance(x, list)

        F,A1,A2,mask = x

        mask_resized = tf.image.resize_images(
            mask,
            size=[tf.shape(A1)[1], tf.shape(A1)[2]],  # 目标尺寸 [H, W]
            method=tf.image.ResizeMethod.BILINEAR  # 插值方法
        )
        mask_proj = Conv2D(1, 15, padding='same')(mask_resized)
        mask_proj = tf.tile(mask_proj, [1, 1, 1, tf.shape(A1)[-1]])

        one = tf.ones_like(A2)
        zero = tf.zeros_like(A2)
        AA2 = tf.where(((A1+A2)/2) < mask_proj, x=zero, y=one)
        F = tf.multiply(F,AA2)
        return F

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        image_size = input_shape[0][1]
        input_dim = (input_shape[0][0],image_size,image_size, 1 * input_shape[0][3])
        return input_dim



def AMFRS(img_rows, img_cols, num_PC, nb_classes):
    CNNInput = Input(shape=(img_rows, img_cols, num_PC), name='i0')

    # Reshape to 4D tensor
    F = Reshape([img_rows, img_cols, num_PC])(CNNInput)
    F = BatchNormalization()(F)
    F = Dropout(rate=0.3)(F)

    F_it, mask = AMG(mode='background_enhance')(F)

    branch_orig = Conv2D(256, 3, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(F)
    branch_rev = Conv2D(256, 3, strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(F_it)

    F2 = Add()([branch_orig, branch_rev])
    # F2 = Multiply()([branch_orig, branch_rev])
    # F2 = concatenate([branch_orig, branch_rev])

    x = Conv2D(filters=256, kernel_size=3, activation='relu', strides=2, padding='same', kernel_regularizer=regularizers.l2(0.01))(F2)

    x = MaxPooling2D(3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=256, kernel_size=3, activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)


    x = Conv2D(filters=256, kernel_size=3, activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    x_A = BatchNormalization()(x)
    x_A = Dropout(rate=0.3)(x_A)

    # Attention blocks
    x1_l_a = attention_vertical_block(x_A)
    x1_l_b = attention_horizontal_block(x_A)
    x_f1, x_f2 = FR()([x, x1_l_a, x1_l_b, mask])

    x1 = concatenate([x_A, x_f1])
    x1 = BatchNormalization()(x1)


    # More attention blocks
    x = Conv2D(filters=256, kernel_size=3, activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x1_l_a = attention_vertical_block(x)
    x1_l_b = attention_horizontal_block(x)
    x2 = FS()([x, x1_l_b, x1_l_a, mask])  # ab
    x2 = BatchNormalization()(x2)

    x = concatenate([x2, x1])  # concatenate

    # Flatten and dense layers
    x = Flatten()(x)
    F = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    F = Dropout(rate=0.3)(F)

    # Final classification layer
    n = math.ceil(math.sqrt(K.int_shape(F)[-1]))
    F = Dense(nb_classes, activation='softmax', name='classifier', kernel_initializer=Symmetry(n=n, c=nb_classes))(F)
    model = Model(inputs=[CNNInput], outputs=[F])

    return model

