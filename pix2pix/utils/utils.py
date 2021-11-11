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

def separate_image(image):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    original_w, original_h = tf.shape(image)
    input_w = original_w // 2
    input_img = image[:, input_w:, :]
    target_img = image[:, :input_w, :]
    return input_img, target_img
    
def preprocess(image, resize_width=286, resize_height=286, image_width=256, image_height=256, mode="train"):
    if mode == "train":
        image = tf.image.resize(image,size=(resize_width, resize_height),
                                   method=tf.image.ResizeMethod.BILINEAR)
        image = tf.image.random_crop(image,size=(image_width,image_height,3))
    image = tf.cast(hr_image,tf.float32)/127.5 - 1
    image = tf.cast(image,tf.float32)
    return image
