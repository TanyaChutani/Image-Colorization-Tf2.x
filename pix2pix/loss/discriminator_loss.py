import tensorflow as tf

class Discriminator_Loss(tf.keras.losses.Loss):
    def __init__(self,**kwargs):
        super(Generator_Loss,self).__init__()
        self.mse_loss = tf.keras.losses.MeanSquaredError()

    def call(self, disc_true, disc_op):
        real_loss = self.mse_loss(disc_true, tf.ones_like(disc_true))
        fake_loss = self.mse_loss(disc_op, tf.zeros_like(disc_op))
        final_disc_loss = real_loss + fake_loss
        return final_disc_loss
    
