import tensorflow as tf

from pix2pix.utils.utils import train_preprocess_image, test_preprocess_image

def data_generator(path, batch_size, mode="train"):
    dataset = tf.data.Dataset.list_files(path,tf.data.experimental.AUTOTUNE)
    preprocess = train_preprocess_image if mode == "train" else test_preprocess_image
    dataset = dataset.map(preprocess,tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

