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

def separate_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    original_w = tf.shape(image)[1]
    input_w = original_w // 2
    input_img = image[:, input_w:, :]
    target_img = image[:, :input_w, :]
    return input_img, target_img
