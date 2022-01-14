import os
import numpy as np
import tensorflow as tf

class DataGenerator:
    def __init__(
        self,
        mode,
        data_path,
        batch_size,
        resize_dim,
        shuffle=True,
        n_channels=3,
    ):
        self.mode = mode
        self.data_path = data_path
        self.batch_size = batch_size
        self.resize_dim = resize_dim
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.data = os.listdir((os.path.join(self.data_path, self.mode)))
        self.index = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __len__(self):
        return np.ceil(len(self.data) / self.batch_size)

    def __call__(self):
        for i in self.index:
            x, y = self.load(os.path.join(self.data_path, self.mode, self.data[i]))
            yield x, y

    def preprocess(self, image):
        if self.mode == "train":
            image = tf.image.random_crop(
                image, size=(self.resize_dim[0], self.resize_dim[1], 3)
            )
        image = tf.image.resize(
            image, size=self.resize_dim, method=tf.image.ResizeMethod.BILINEAR
        )

        image = image / 255.0
        image = tf.cast(image, tf.float32)
        return image

    def load(self, image_path):
        image = tf.io.read_file(image_path)
        colored_image = tf.image.decode_png(image, channels=self.n_channels)
        colored_image.set_shape([None, None, self.n_channels])

        colored_image = self.preprocess(colored_image)
        gray_image = tf.image.rgb_to_grayscale(colored_image)

        return gray_image, colored_image
