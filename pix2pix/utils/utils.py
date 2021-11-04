import tensorflow as tf

class Reflection_Pad(tf.keras.layers.Layer):
    def __init__(self, pad):
        super(Reflection_Pad,self).__init__()
        self.pad = tuple(pad)

    def call(self, input_tensor):
        pad_width, pad_height = self.pad
        return tf.pad(input_tensor, 
                      [[0, 0],
                       [pad_width, pad_height],
                       [pad_width, pad_height],
                       [0, 0]],
                      mode = "REFLECT")
