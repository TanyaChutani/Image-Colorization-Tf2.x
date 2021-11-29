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
