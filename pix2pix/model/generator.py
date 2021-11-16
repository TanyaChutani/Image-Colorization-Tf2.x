import tensorflow as tf

from pix2pix.utils.utils import Reflection_Pad
from pix2pix.utils.generator_utils import Conv_Encoder_Block, Conv_Decoder_Block

class Generator(tf.keras.layers.Layer):
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
        self.model_output =  Conv_Decoder_Block(3,
                                                False,
                                                self.dropout,
                                                True)
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

    
    def call(self, x, training=None):
        encoder_block = self.make_encoder_block()
        decoder_block = self.make_decoder_block()

        reverse_encode = []
        for encode in encoder_block:
            encode = tf.keras.Sequential(encode)
            x = encode(x)
            reverse_encode.append(x)

        reverse_encoder = reversed(reverse_encode[:-1])
        
        for encode, decode in zip(decoder_block, reverse_encoder):
            encode = tf.keras.Sequential(encode)
            x = encode(x)
            x = tf.keras.layers.Concatenate()([x, decode])
    
        x = self.model_output(x)
        return x
