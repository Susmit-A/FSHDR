import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

class SDC(Layer):
    def __init__(self, filters, kernel_size=(3, 3), padding='same', *args, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters, self.kernel_size, padding='same', activation=LeakyReLU())
        self.conv2 = Conv2D(self.filters, self.kernel_size, padding='same', activation=LeakyReLU(), dilation_rate=(2, 2))
        self.conv3 = Conv2D(self.filters, self.kernel_size, padding='same', activation=LeakyReLU(), dilation_rate=(3, 3))
        self.concat = Concatenate()

    def call(self, inputs, **kwargs):
        x = inputs
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x = self.concat([x1, x2, x3])
        return x

class DRDB_Unit(Layer):
    def __init__(self, growth_rate=32, kernel_size=(3, 3), *args, **kwargs):
        self.kernel_size = kernel_size
        self.growth = growth_rate
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        kernel_size = self.kernel_size
        input_shape = input_shape.as_list()
        self.dilated_conv1_kernel = self.add_weight(name='DRDB_conv1_kernel',
                                         shape=(kernel_size[0], kernel_size[1], input_shape[-1],
                                         self.growth), dtype = tf.float32, initializer = 'glorot_uniform',
                                         trainable = True)
        self.dilated_conv2_kernel = self.add_weight(name='DRDB_conv2_kernel',
                                         shape=(kernel_size[0], kernel_size[1], input_shape[-1] + self.growth,
                                         self.growth), dtype = tf.float32, initializer = 'glorot_uniform',
                                         trainable = True)
        self.dilated_conv3_kernel = self.add_weight(name='DRDB_conv3_kernel',
                                         shape=(kernel_size[0], kernel_size[1], input_shape[-1] + self.growth*2,
                                         self.growth), dtype = tf.float32, initializer = 'glorot_uniform',
                                         trainable = True)
        self.dilated_conv4_kernel = self.add_weight(name='DRDB_conv4_kernel',
                                         shape=(kernel_size[0], kernel_size[1], input_shape[-1] + self.growth*3,
                                         self.growth), dtype = tf.float32, initializer = 'glorot_uniform',
                                         trainable = True)
        self.dilated_conv5_kernel = self.add_weight(name='DRDB_conv5_kernel',
                                         shape=(kernel_size[0], kernel_size[1], input_shape[-1] + self.growth*4,
                                         self.growth), dtype = tf.float32, initializer = 'glorot_uniform',
                                         trainable = True)
        self.dilated_conv6_kernel = self.add_weight(name='DRDB_conv6_kernel',
                                         shape=(kernel_size[0], kernel_size[1], input_shape[-1] + self.growth*5,
                                         self.growth), dtype = tf.float32, initializer = 'glorot_uniform',
                                         trainable = True)
        self.conv7_kernel = self.add_weight(name='DRDB_conv7_kernel',
                                         shape=(1, 1, input_shape[-1] + self.growth*6, input_shape[-1]),
                                         dtype = tf.float32, initializer = 'glorot_uniform', trainable = True)
                                         
        self.dilated_conv1_bias = self.add_weight(name='DRDB_conv1_bias', shape=(self.growth,), dtype=tf.float32,
                                                  initializer = 'zeros', trainable = True)
        self.dilated_conv2_bias = self.add_weight(name='DRDB_conv2_bias', shape=(self.growth,), dtype=tf.float32,
                                                  initializer = 'zeros', trainable = True)
        self.dilated_conv3_bias = self.add_weight(name='DRDB_conv3_bias', shape=(self.growth,), dtype=tf.float32,
                                                  initializer = 'zeros', trainable = True)
        self.dilated_conv4_bias = self.add_weight(name='DRDB_conv4_bias', shape=(self.growth,), dtype=tf.float32,
                                                  initializer = 'zeros', trainable = True)
        self.dilated_conv5_bias = self.add_weight(name='DRDB_conv5_bias', shape=(self.growth,), dtype=tf.float32,
                                                  initializer = 'zeros', trainable = True)
        self.dilated_conv6_bias = self.add_weight(name='DRDB_conv6_bias', shape=(self.growth,), dtype=tf.float32,
                                                  initializer = 'zeros', trainable = True)
        self.conv7_bias = self.add_weight(name='DRDB_conv7_bias', shape=(input_shape[-1],), dtype=tf.float32,
                                                  initializer = 'zeros', trainable = True)
                                         
    def compute_output_shape(self, input_shape):
        shape = input_shape.as_list()
        return tf.TensorShape((input_shape[0], input_shape[1], input_shape[2], input_shape[3]))
        
    def call(self, inputs, **kwargs):
        stride = (1, 1, 1, 1)
        x1_ = inputs
        x1 = tf.nn.conv2d(x1_, self.dilated_conv1_kernel, strides=stride, padding='SAME', dilations=(1, 2, 2, 1)) + self.dilated_conv1_bias
        x1 = tf.nn.relu(x1)
        
        x2_ = tf.concat([x1_, x1], axis=-1)
        x2 = tf.nn.conv2d(x2_, self.dilated_conv2_kernel, strides=stride, padding='SAME', dilations=(1, 2, 2, 1)) + self.dilated_conv2_bias
        x2 = tf.nn.relu(x2)
        
        x3_ = tf.concat([x2_, x2], axis=-1)
        x3 = tf.nn.conv2d(x3_, self.dilated_conv3_kernel, strides=stride, padding='SAME', dilations=(1, 2, 2, 1)) + self.dilated_conv3_bias
        x3 = tf.nn.relu(x3)
        
        x4_ = tf.concat([x3_, x3], axis=-1)
        x4 = tf.nn.conv2d(x4_, self.dilated_conv4_kernel, strides=stride, padding='SAME', dilations=(1, 2, 2, 1)) + self.dilated_conv4_bias
        x4 = tf.nn.relu(x4)
        
        x5_ = tf.concat([x4_, x4], axis=-1)
        x5 = tf.nn.conv2d(x5_, self.dilated_conv5_kernel, strides=stride, padding='SAME', dilations=(1, 2, 2, 1)) + self.dilated_conv5_bias
        x5 = tf.nn.relu(x5)
        
        x6_ = tf.concat([x5_, x5], axis=-1)
        x6 = tf.nn.conv2d(x6_, self.dilated_conv6_kernel, strides=stride, padding='SAME', dilations=(1, 2, 2, 1)) + self.dilated_conv6_bias
        x6 = tf.nn.relu(x6)

        x7_ = tf.concat([x6_, x6], axis=-1)
        x7 = tf.nn.conv2d(x7_, self.conv7_kernel, strides=stride, padding='VALID', dilations=(1, 1, 1, 1)) + self.conv7_bias
        out = x7 + x1_
        return out

class Divider(Layer):
    def __init__(self):
        super().__init__()
    def call(self, inputs):
        return inputs[0] / (inputs[1] + 1e-7)

class StridedConv(Layer):
    def __init__(self, channels=64, kernel_size=5, stride=2):
        super().__init__()
        self.c = channels
        self.ks = kernel_size
        self.s = (1, stride, stride, 1)
    
    def build(self, input_shape):
        input_shape = input_shape.as_list()
        self.kernel = self.add_weight(name='StridedConv_kernel',
                                      shape=(self.ks, self.ks, input_shape[-1], self.c), dtype = tf.float32,
                                      initializer = 'glorot_uniform', trainable = True)
        self.bias = self.add_weight(name='StridedConv_bias', shape=(self.c,), dtype=tf.float32,
                                    initializer = 'zeros', trainable = True)
    
    def compute_output_shape(self, input_shape):
        shape = input_shape.as_list()
        return tf.TensorShape((input_shape[0], input_shape[1], input_shape[2], self.c))
    
    def call(self, inputs):
        h, w = inputs.shape[1:3]
        if h is None or (h % self.s[1] == 0):
            pad_h = max(self.ks - self.s[1], 0)
        else:
            pad_h = max(self.ks - (h % self.s[1]), 0)
        if w is None or (w % self.s[2] == 0):
            pad_w = max(self.ks - self.s[2], 0)
        else:
            pad_w = max(self.ks - (w % self.s[2]), 0)
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l
        inputs = tf.pad(inputs, [[0, 0], [pad_t, pad_b], [pad_l, pad_r], [0, 0]], "REFLECT")
        out = tf.nn.conv2d(inputs, self.kernel, strides=self.s, padding='VALID', dilations=(1, 1, 1, 1)) + self.bias
        return out