import tensorflow as tf

from pix2pix.model.generator import Generator
from pix2pix.model.discriminator import Discriminator
from pix2pix.loss.discriminator_loss import Discriminator_Loss
from pix2pix.loss.generator_loss import Generator_Loss


class PIX2PIX(tf.keras.models.Model):
    def __init__(self,
                 var_lambda=100,
                 **kwargs):
        super(PIX2PIX, self).__init__()
        self.generator_model = Generator()
        self.discriminator_model = Discriminator()
        self.gen_loss = Generator_Loss()
        self.disc_loss = Discriminator_Loss()
        self.var_lambda = var_lambda
        
    def compile(self, generator_optimizer, discriminator_optimizer, loss):
        super(PIX2PIX, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.l1_loss = loss

    def train_step(self, data):
        input_image, output_image = data

        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            gen_output = self.generator_model(input_image, training=True)
            disc_real_output = self.discriminator_model(input_image, output_image, training=True)
            disc_fake_output = self.discriminator_model(input_image, gen_output, training=True)
            
            generator_pixel_loss = self.l1_loss(gen_output, output_image)

            generator_loss = self.gen_loss(disc_fake_output,
                                           gen_output)

            final_gen_loss = generator_loss + self.var_lambda * generator_pixel_loss    

            discriminator_loss = self.disc_loss(disc_real_output, disc_fake_output)


        discriminator_gradients = discriminator_tape.gradient(discriminator_loss,
                                                              self.discriminator_model.trainable_variables)
        generator_gradients = generator_tape.gradient(generator_loss, self.generator_model.trainable_variables)

        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator_model.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator_model.trainable_variables))

        return {"generator_loss": final_gen_loss,
                "discriminator_loss": discriminator_loss}
    
    
