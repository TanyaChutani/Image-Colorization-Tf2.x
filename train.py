import tensorflow as tf
import argparse
import sys

from pix2pix.model.generator import Generator
from pix2pix.model.discriminator import Discriminator
from pix2pix.model.pix2pix import PIX2PIX
from pix2pix.data.data_generator import data_generator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gan_epoch',
                        '--epochs',
                        type=int,
                        metavar='',
                        default = 200)

    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        metavar='',
                        default = 16)

    parser.add_argument('-img_dir',
                        '--train_path',
                        type=str,
                        metavar='',
                        default = '/pix2pix_tf/cityscapes/train/*.jpg')

    parser.add_argument('-w',
                        '--gan_weights_path',
                        type=str,
                        metavar='',
                        default = '/pix2pix_tf/weights/')
    args = parser.parse_args()
    return args
