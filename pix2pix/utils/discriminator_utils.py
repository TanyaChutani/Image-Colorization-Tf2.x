import tensorflow as tf

from pix2pix.utils.utils import Reflection_Pad


class Conv_Downsample_Block(tf.keras.layers.Layer):
    def __init__(self, filters, stride, bn, last_layer=False):
        super(Conv_Downsample_Block, self).__init__()
        self.filters = filters
        self.stride = stride
        self.bn = bn
        self.last_layer = last_layer
        self.conv_layer = tf.keras.layers.Conv2D(
            self.filters, 4, self.stride, padding="same"
        )
        self.conv_last_layer = tf.keras.layers.Conv2D(self.filters, 4, self.stride)
        self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.8)
        self.lrelu_layer = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.reflection_pad = Reflection_Pad(pad=(1, 1))

    def call(self, input_tensor, training=None):
        if self.last_layer:
            x = self.conv_last_layer(input_tensor)
            x = self.reflection_pad(x)
        else:
            x = self.conv_layer(input_tensor)
        if self.bn:
            x = self.bn_layer(x, training=training)
        x = self.lrelu_layer(x)
        return x
