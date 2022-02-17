import argparse
import tensorflow as tf

from pix2pix.loss.metric import psnr
from pix2pix.model.pix2pix import PIX2PIX
from pix2pix.model.generator import Generator
from pix2pix.model.discriminator import Discriminator
from pix2pix.data.data_generator import DataGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gan_epoch", "--epochs", type=int, metavar="", default=100)

    parser.add_argument("-b", "--batch_size", type=int, metavar="", default=16)

    parser.add_argument("-ip_size", "--input_size", type=int, metavar="", default=1024)

    parser.add_argument(
        "-img_dir",
        "--data_path",
        type=str,
        metavar="",
        default="/home/ubuntu/pix2pix_data",
    )

    parser.add_argument(
        "-w",
        "--gan_weights_path",
        type=str,
        metavar="",
        default="/home/ubuntu/pix2pix_tf/weights/",
    )
    parser.add_argument(
        "-gen_w",
        "--generator_weights_path",
        type=str,
        metavar="",
        default="/home/ubuntu/pix2pix_tf/gen_weights/",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    train_gen = DataGenerator(
        mode="train",
        data_path=args.data_path,
        batch_size=args.batch_size,
        resize_dim=(args.input_size, args.input_size),
    )
    val_gen = DataGenerator(
        mode="valid",
        data_path=args.data_path,
        batch_size=args.batch_size,
        resize_dim=(args.input_size, args.input_size),
    )
    train_dataset = tf.data.Dataset.from_generator(
        train_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            [args.input_size, args.input_size, 1],
            [args.input_size, args.input_size, 3],
        ),
    )
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.repeat()
    
    val_dataset = tf.data.Dataset.from_generator(
        val_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            [args.input_size, args.input_size, 1],
            [args.input_size, args.input_size, 3],
        ),
    )
    val_dataset = val_dataset.batch(args.batch_size)
    val_dataset = val_dataset.repeat()
    
    learning_rate = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[args.epochs / 2], values=[0.0001, 0.00001]
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.gan_weights_path,
            monitor="total_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
        )
    ]

    model = PIX2PIX()
    model.compile(
        generator_optimizer=tf.keras.optimizers.Adam(learning_rate),
        discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate),
        metric=psnr,
    )
    model.fit(
        train_dataset,
        steps_per_epoch=train_gen.__len__(),
        epochs=args.epochs,
        validation_data=val_dataset,
        validation_steps=val_gen.__len__(),
        callbacks=callbacks,
    )
    model.generator_model.save_weights(args.generator_weights_path)


if __name__ == "__main__":
    main()
