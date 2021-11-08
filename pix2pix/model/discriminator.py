import tensorflow as tf

from pix2pix.utils.utils import Reflection_Pad
from pix2pix.utils.discriminator_utils import Conv_Downsample_Block

class Discriminator(tf.keras.layers.Layer):
    def __init__(self, filters=64, 
                 stride=2,
                 bn=True):
        super(Discriminator, self).__init__()
        self.filters = filters
        self.stride = stride
        self.bn = bn
        self.make_discriminator_blocks = self.make_discriminator_block()
        self.reflection_pad = Reflection_Pad(pad=(1,1))
        self.model_output = tf.keras.layers.Conv2D(1,
                                                   4,
                                                   1,
                                                   padding='valid')

    def make_discriminator_block(self):
        label = []
        label.append(Conv_Downsample_Block(self.filters, 
                                         self.stride,
                                         bn=False))
        for _ in range(2):
            self.filters = self.filters * 2
            label.append(Conv_Downsample_Block(self.filters,
                                             self.stride,
                                             self.bn))
    
        self.filters = self.filters * 2
        label.append(Conv_Downsample_Block(filters=self.filters,
                                           stride=1, 
                                           bn=self.bn))
        return tf.keras.Sequential(label)

    def call(self, image_1, image_2, training=None):
        x = tf.concat([image_1, image_2], axis=3)
        x = self.make_discriminator_blocks(x, training=training)
        x = self.reflection_pad(x)
        x = self.model_output(x)
        return x
