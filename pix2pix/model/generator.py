import tensorflow as tf

from pix2pix.utils.utils import Reflection_Pad
from pix2pix.utils.generator_utils import Conv_Encoder_Block, Conv_Decoder_Block


class Generator(tf.keras.Model):
    def __init__(
        self,
        bn=True,
        relu=False,
        tanh=False,
        filters=64,
        dropout=False,
    ):
        super(Generator, self).__init__()
        self.bn = bn
        self.relu = relu
        self.tanh = tanh
        self.dropout = dropout
        self.filters = filters

        self.conv_layer = tf.keras.layers.Conv2DTranspose(
            3, 4, 2, padding="same", use_bias=False
        )
        self.concat_layer = tf.keras.layers.Concatenate()

        self.decoder_block = self.make_decoder_block()
        self.encoder_block = self.make_encoder_block()

    def make_decoder_block(self):
        label = []
        label.append(Conv_Decoder_Block(self.filters, False, self.relu))
        for _ in range(2):
            self.filters = self.filters * 2
            label.append(Conv_Decoder_Block(self.filters, self.bn, self.relu))
        self.filters = self.filters * 2
        for _ in range(5):
            label.append(Conv_Decoder_Block(self.filters, self.bn, self.relu))
        return label

    def make_encoder_block(self):
        label = []
        for _ in range(3):
            label.append(Conv_Encoder_Block(self.filters, self.bn, True, self.tanh))
        label.append(Conv_Encoder_Block(self.filters, self.bn, self.dropout, self.tanh))
        for _ in range(3):
            self.filters = int(self.filters / 2)
            label.append(
                Conv_Encoder_Block(self.filters, self.bn, self.dropout, self.tanh)
            )
        return label

    def call(self, input_tensor, training=None):
        x1 = self.decoder_block[0](input_tensor, training=training)
        x2 = self.decoder_block[1](x1, training=training)
        x3 = self.decoder_block[2](x2, training=training)
        x4 = self.decoder_block[3](x3, training=training)
        x5 = self.decoder_block[4](x4, training=training)
        x6 = self.decoder_block[5](x5, training=training)
        x7 = self.decoder_block[6](x6, training=training)

        backbone = self.decoder_block[7](x7, training=training)

        x8 = self.encoder_block[0](backbone, training=training)
        x = self.concat_layer([x8, x7])
        x9 = self.encoder_block[1](x, training=training)
        x = self.concat_layer([x9, x6])
        x10 = self.encoder_block[2](x, training=training)
        x = self.concat_layer([x10, x5])
        x11 = self.encoder_block[3](x, training=training)
        x = self.concat_layer([x11, x4])
        x12 = self.encoder_block[4](x, training=training)
        x = self.concat_layer([x12, x3])
        x13 = self.encoder_block[5](x, training=training)
        x = self.concat_layer([x13, x2])
        x14 = self.encoder_block[6](x, training=training)
        x = self.concat_layer([x14, x1])
        x15 = self.conv_layer(x)
        return x15
