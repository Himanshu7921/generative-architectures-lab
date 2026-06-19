# Neural Style Transfer Experiments

This notebook contains my experiments and implementation of **"A Neural Algorithm of Artistic Style"** by Gatys et al.

The goal is to understand how convolutional neural networks represent **content** and **style**, and how these representations can be combined to generate artistic images.

In this notebook, I will:

* Explore feature representations learned by CNNs.
* Visualize activations and image reconstructions from intermediate layers.
* Implement content and style losses from the original paper.
* Compute and analyze Gram Matrices for style representation.
* Perform Neural Style Transfer using the official VGG-19 model.
* Experiment with a custom VGG-19 trained on Tiny ImageNet.
* Compare feature representations between pretrained and custom models.
* Investigate how different layers influence content and style generation.

The primary focus is learning, experimentation, and developing an intuition for CNN feature representations through Neural Style Transfer.
![alt text](image.png)