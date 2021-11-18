import tensorflow as tf

from pix2pix.model.generator import Generator
from pix2pix.model.discriminator import Discriminator
from pix2pix.loss.discriminator_loss import Discriminator_Loss
from pix2pix.loss.generator_loss import Generator_Loss


class PIX2PIX(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(PIX2PIX, self).__init__()
        self.generator_model = Generator()
        self.discriminator_model = Discriminator()
        self.gen_loss = Generator_Loss()
        self.disc_loss = Discriminator_Loss()
    
    def compile(self, generator_optimizer, discriminator_optimizer, loss_fn, psnr_metric):
        super(PIX2PIX, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_fn = loss_fn
        self.psnr_metric = psnr_metric
