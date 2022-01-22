import tensorflow as tf

from pix2pix.utils.utils import Reflection_Pad


class Conv_Decoder_Block(tf.keras.layers.Layer):
    def __init__(self, filters, bn, relu):
        super(Conv_Decoder_Block, self).__init__()
        self.filters = filters
        self.relu = relu
        self.bn = bn
        self.conv_layer = tf.keras.layers.Conv2D(self.filters, 4, 2, padding="same")
        self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.8)
        self.lrelu_layer = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.reflection_pad = Reflection_Pad(pad=(1, 1))
        self.relu_layer = tf.keras.layers.ReLU()

    def call(self, input_tensor, training=None):
        x = self.conv_layer(input_tensor)
        if self.bn:
            x = self.bn_layer(x, training=training)
        x = self.relu_layer(x) if self.relu else self.lrelu_layer(x)
        return x


class Conv_Encoder_Block(tf.keras.layers.Layer):
    def __init__(self, filters, bn, dropout, tanh):
        super(Conv_Encoder_Block, self).__init__()
        self.filters = filters
        self.bn = bn
        self.tanh = tanh
        self.dropout = dropout
        self.conv_layer = tf.keras.layers.Conv2DTranspose(
            self.filters, 4, 2, padding="same", use_bias=False
        )
        self.bn_layer = tf.keras.layers.BatchNormalization()
        self.relu_layer = tf.keras.layers.ReLU()
        self.tanh_layer = tf.keras.layers.Activation(tf.nn.tanh)
        self.dropout_layer = tf.keras.layers.Dropout(0.5)

    def call(self, input_tensor, training=None):
        x = self.conv_layer(input_tensor)
        if self.bn:
            x = self.bn_layer(x, training=training)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.tanh_layer(x) if self.tanh else self.relu_layer(x)
        return x
