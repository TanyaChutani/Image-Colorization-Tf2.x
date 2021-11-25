import tensorflow as tf

from pix2pix.utils.utils import Reflection_Pad
from pix2pix.utils.generator_utils import Conv_Encoder_Block, Conv_Decoder_Block

class Generator(tf.keras.Model):
    def __init__(self,filters=64, 
                 bn=True,
                 relu=False,
                 tanh=False,
                 dropout=False
                ):
        super(Generator, self).__init__()
        self.filters = filters
        self.bn = bn
        self.relu = relu
        self.tanh = tanh
        self.dropout = dropout 
        self.conv_layer = tf.keras.layers.Conv2DTranspose(3, 
                                                          4, 
                                                          2,
                                                          padding='same',
                                                          use_bias=False,
                                                          activation="tanh")
        self.encoder_block = self.make_encoder_block()
        self.decoder_block = self.make_decoder_block()

    def make_encoder_block(self):
        label = []
        label.append(Conv_Encoder_Block(self.filters, 
                                        False,
                                        self.relu))
        for _ in range(2):
            self.filters = self.filters * 2
            label.append(Conv_Encoder_Block(self.filters,
                                            self.bn,
                                            self.relu))
        self.filters = self.filters * 2
        for _ in range(4):
            label.append(Conv_Encoder_Block(self.filters,
                                            self.bn,
                                            self.relu))
        label.append(Conv_Encoder_Block(self.filters,
                                        self.bn, 
                                        True))
        return label
    
    def make_decoder_block(self):
        label = []
        for _ in range(3):
            label.append(Conv_Decoder_Block(self.filters,
                                            self.bn,
                                            True,
                                            self.tanh))
        label.append(Conv_Decoder_Block(self.filters,
                                        self.bn,
                                        self.dropout,
                                        self.tanh))
        for _ in range(3):
            self.filters = int(self.filters / 2)
            label.append(Conv_Decoder_Block(self.filters,
                                            self.bn,
                                            self.dropout,
                                            self.tanh))    
        return label

    
    def call(self, x, training):
        x1 = self.encoder_block[0](x)
        x2 = self.encoder_block[1](x1)
        x3 = self.encoder_block[2](x2)
        x4 = self.encoder_block[3](x3)
        x5 = self.encoder_block[4](x4)
        x6 = self.encoder_block[5](x5)
        x7 = self.encoder_block[6](x6)
        
        backbone = self.encoder_block[7](x7)
    
        x8 = self.decoder_block[0](backbone, x7)
        x9 = self.decoder_block[1](x8, x6)
        x10 = self.decoder_block[2](x9, x5)
        x11 = self.decoder_block[3](x10, x4)
        x12 = self.decoder_block[4](x11, x3)
        x13 = self.decoder_block[5](x12, x2)
        x14 = self.decoder_block[6](x13, x1)
        x15 = self.conv_layer(x14)
        
        return x15
