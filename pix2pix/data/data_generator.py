import os
import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path, 
                 mode = "train", 
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
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.data):
            self.batch_size = len(self.data) - index * self.batch_size
        batch = self.data[index * self.batch_size : (index + 1) * self.batch_size]
        x, y = self.data_generation(batch)
        return x, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def preprocess(self, image):
        if self.mode == "train":
            image = tf.image.resize(image,size=self.resize_dim,
                                       method=tf.image.ResizeMethod.BILINEAR)
            image = tf.image.random_crop(image,size=(self.dim[0],self.dim[1],3))
        image = tf.image.resize(image, size=self.dim)
        image = image/127.5 - 1
        image = tf.cast(image,tf.float32)
        return image

    def load(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=self.n_channels)
        image.set_shape([None, None, 3])
        original_w = tf.shape(image)[1]
        input_w = original_w // 2
        input_img = image[:, input_w:, :]
        target_img = image[:, :input_w, :]
        
        input_image = self.preprocess(input_img)
        target_image = self.preprocess(target_img)

        return input_image, target_image

    def data_generation(self, batch):
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))
        for i, batch_id in enumerate(batch):

            x[i,], y[i,] = self.load(
                os.path.join(self.data_path, batch_id),
            )
        return x, y
