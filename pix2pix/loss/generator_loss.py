import tensorflow as tf


class Generator_Loss(tf.keras.losses.Loss):
    def __init__(self):
        super(Generator_Loss, self).__init__()
        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, disc_op, gen_op):
        gen_loss = self.bce_loss(tf.ones_like(disc_op), disc_op)
        return gen_loss
