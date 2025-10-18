# Pallets

![A cute robot staring at blocks laying on the floor. The background is made of pieces of wood stained in the different bright colors](pallets.jpg)

Pallets is a project to explore pixel art generation in neural networks using toy model building blocks.


## Abstract

We explore the application of convolutional autoencoders to generate pixel art. Our approach is twofold: first, we implement a standard RGB-based autoencoder model to understand baseline performance and highlight issues with current approaches - including those using very large generative models.  Second, we introduce a novel one-hot encoded color mapping autoencoder designed to adhere strictly to predefined color palettes, a critical aspect of pixel art.

We demonstrate a dimensionality reduction of the problem alongside order of magnitude performance improvement relative to the aesthetic of pixel art. We then suggest that this can be more broadly applied on larger images, more complex model architectures and ambitious generative pieces. 


## Notebooks

### The Dataset

* [Introduction to the Images](nb/dataset/IntroToDSImages.ipynb): Explore the image side of dataset and how we represent it in the models.
* [Earrings Label](nb/dataset/FindEarrings.ipynb): Address issues with the earrings label.

### Autoencoders

* [AE](nb/ae/AE.ipynb): A basic autoencoder for images with 4 color channels for RGBA
* [ConvAE](nb/ae/ConvAE.ipynb): A convolutionary form of AE
* [AEOneHot](nb/ae/AEOneHot.ipynb): A basic autoencoder for images with a one-hot encoded representation of each unique color in the dataset
* [ConvAEOneHot](nb/ae/ConvAEOneHot.ipynb): A convolutionary form of AEOneHot

### Variational Autoencoders

* [VAE](nb/vae/VAE.ipynb): A basic variational autoencoder for one-hot encoded images
* [ConvVAE](nb/vae/ConvVAE.ipynb): A convolutionary form of VAE
* [LabeledVAE](nb/vae/LabeledVAE.ipynb): A labeled form of VAE
* [LabeledConvVae](nb/vae/LabeledConvVAE.ipynb): A combination convolutionary & labeled form of VAE

### Conditional Variational Autoencoders

* [CVAE](nb/cvae/CVAE.ipynb): A basic conditional variational autoencoder for one-hot encoded images
* [ConvCVAE](nb/cvae/ConvCVAE.ipynb): A convolutionary form of VAE
* [LabeledCVAE](nb/cvae/LabeledCVAE.ipynb): A labeled form of VAE
* [LabeledConvCVae](nb/cvae/LabeledConvCVAE.ipynb): A combination convolutionary & labeled form of VAE

### Gumbel Softmax

* [GSVAE](nb/gumbel/GSVAE.ipynb): A basic variational autoencoder with gumbel softmax reparameterization
* [LabeledGSVAE](nb/gumbel/LabeledGSVAE.ipynb): A labeled form of GSVAE

### Math Visualizations

* [Visualizing Convolutions](nb/mathviz/VisualizingConvolutions.ipynb): Applies different types of convolutions to graphical representations of an X and an O, and displays an image for every layer of math applied.
* [Notes On The Simplex](nb/mathviz/NotesOnTheSimplex.ipynb): A visual representation of a simplex, the structure that underlines Gumbel Softmax.
* [Using Gumbel Softmax](nb/mathviz/UsingGumbelSoftmax.ipynb): An implementation of Eric Jang's gumbel softax (_one of the original authors of Gumbel Softmax_) is compared with the implementation packaged with PyTorch.


## Setup

Get both the cpunks-10k repo and this one

```shell
git clone https://github.com/jmsdnns/cpunks
git clone https://github.com/jmsdnns/pallets
```

Then setup a venv for pallets and install it

```
cd pallets
python -mvenv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Credits

A huge shoutout to [@tnn1t1s](https://github.com/tnn1t1s) for starting this project and for mentoring me as I study machine learning.

