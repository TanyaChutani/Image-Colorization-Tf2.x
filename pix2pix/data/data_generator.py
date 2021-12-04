import os
import numpy as np
import tensorflow as tf

class DataGenerator():
    def __init__(self, data_path, 
                 mode="train", 
                 batch_size=8, 
                 dim=(256, 256), 
                 n_channels=3, 
                 shuffle=True, 
                 resize_dim=(286, 286)
        ):
        self.dim = dim
        self.mode = mode
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.resize_dim = resize_dim  
        self.data_path = data_path
        self.data =  os.listdir(data_path)
        self.index = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __len__(self):
        return (np.ceil(len(self.data) / self.batch_size))

    def __call__(self):
        for i in (self.index):
            x, y = self.load(os.path.join(self.data_path, 
                                          self.data[i]))
            yield x, y
    
    def preprocess(self, image):
        if self.mode == "train":
            image = tf.image.resize(image,size=self.resize_dim,
                                       method=tf.image.ResizeMethod.BILINEAR)
            image = tf.image.random_crop(image,size=(self.dim[0],self.dim[1],3))
        image = image/255.0
        image = tf.cast(image,tf.float32)
        return image

    def load(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=self.n_channels)
        image.set_shape([None, None, 3])
        original_w = tf.shape(image)[1]
        input_w = original_w // 2
        input_img = image[:, input_w:, :]
        target_img = image[:, :input_w, :]
        
        input_image = self.preprocess(input_img)
        target_image = self.preprocess(target_img)

        return input_image, target_image
