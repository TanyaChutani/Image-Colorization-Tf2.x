# Pix2pix

A Tensorflow 2.x implementation of Pix2pix GAN.https://arxiv.org/abs/1611.07004

# Architecture
![](assets/paper.png)

### Commands to download the data
```
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip

jar xvf  DIV2K_train_HR.zip
jar xvf DIV2K_valid_HR.zip
```

### Command to train the model
```
pip install -r requirements.txt
python train.py --epochs 200 --batch_size 32 --input_size 256 --data_path pix2pix_data/
```
