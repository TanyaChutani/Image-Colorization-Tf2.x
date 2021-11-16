import tensorflow as tf

from pix2pix.utils.generator_utils import Conv_Encoder_Block, Conv_Decoder_Block
from pix2pix.utils.utils import Reflection_Pad

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
        self.make_encoder_blocks = self.make_encoder_block()
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
    
