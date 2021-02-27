
# Reduced Precision and Binary Neural Networks
## Introduction

This Research project compares the accuracy of AlexNet and GoogleNet for three different precision levels- Binary Precision, 8-Bit Reduced Precision and Full Precision. Keras/ TensorFlow backend is used.
Code is developed based on a [lasagne/theano](https://github.com/MatthieuCourbariaux/BinaryNet) version of [BinaryNet](https://papers.nips.cc/paper/6573-binarized-neural-networks) for Binary Network. For 8-Bit Quantized Network this code is based on [N-Bit Qnn](https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow).

## Installation

Running this code requires:

1. [Tensorflow](https://www.tensorflow.org/install/) version 1.14 or 1.15
2. [Keras 2.0](https://keras.io/) version 2.2.5; other versions (2.3.x) have some known issue with 8-Bit QNN model
3. [Mathplotlib](https://matplotlib.org/) for graphs
3. [Larq](https://larq.dev/) API for Binary Network visualization and layers
4. [Pillow](https://pillow.readthedocs.io/en/stable/) for handling images in the dataset  
5. Dataset- [Kaggle Cat and Dogs](https://www.kaggle.com/tongpython/cat-and-dog) 

## File and code structure
There are two folders- Alexnet and Googlenet. Both of them have 3 main files each for- Full Precision, 8-Bit and Binary Net(BNN) model. Apart from that, each main Net folder has subfolder namely Quantized which have customized Keras layers as per BNN requirements. Also, every main folder has 2 python files namely 'quantized_layers' and 'quantized_ops' which are the backbone for operations happening in 8-Bit quantized model.

## Train and evaluate your Network

This project includes Training and evaluation with Kaggle Cats and Dog dataset. In the code segment, please modify file path for train and test image as per location of your computer or server. Currently, it is the default path used while running codes on the local machine.
Once the path is set you can run individually codes for all three precision for both networks directly from IDE.

__Note:__ Sometime when you run these files on [Google Colab](https://colab.research.google.com/) server or linux system, then if graphs are not plotting or giving error, please change 'acc' to 'accuracy' in plot segment of code and section where 'earlystopping' as callback is declared.

## Additional material: Ruuning code on Google Colab
If you want to run these files on Google Colab server. You can have a look at this link. I am sharing a demo project. Based on this you can copy paste main code of project and run it online. Use this [link](https://colab.research.google.com/drive/1XVq2WrhNnE84U91QCDcFdO0tNTNzJIc1) , this is alexnet-8 bit model. Please note that this is just to give overview regarding how you can run the Deep learning model without much overhead of installing software and hardware consideration. You need kaggle API to download dataset from keggle website if you want to run it on Google Server.

## Credits


https://github.com/BertMoons/QuantizedNeuralNetworks-Keras-Tensorflow
https://github.com/tensorpack/tensorpack/tree/master/examples/DoReFa-Net
