# A simple deep impainting algorithm based on generative adversarial networks (GANs)

### Requirements

* A dataset (we used the Paris StreetView dataset, however, it can not be made publicly available)
* Python 3.6+
* Pytorch 1.0+
* Opencv 3.4.2+

### Instructions

#### Training

Before running the training code (train.py), make sure you have opened it and appropriately changed the variables such as: the path to the dataset, the format of the images in the dataset, the hole size, the path to the folder where the trained model will be saved, the number of training epochs, the learning rates, etc. To run the training, simple execute the following command:

```sh
$ python train.py
```

#### Test

Before testing, make sure you already have trained a model and specify the path to it inside the test.py file. Be attentive to the other variables as the path to the output folder where the reconstructed images will be saved. The test code is run by executing:

```sh
$ python test.py
```

### Troubleshooting

#### Multi-GPU code and batch size

This code was written to take advantage of multiple GPU's and the batch size should be at least equal to the number of GPU's available in your machine.

#### Saving models and images

Make sure the specified directories where you wish to save the trained model (in the train.py code) and the reconstructed images (in the test.py code) exist.

#### The Paris StreetView dataset

The Paris StreetView dataset cannot be made publicly available due to Google's restrictions. Please contact me by email if you need it for experimenting algorithms.

