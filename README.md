# Udacity Self Driving Car Nanodegree
## Semantic Segmentation Project 

### Overview

This is a project for Udacity's Self Driving Car Nanodegree. The objective is to label the pixels of a road in images using a Fully Convolutional Network (FCN).

---
### Video Demo

<img src="/images/video.gif" width="640">

---
### Background

This project is based on [FCN-8 architecture](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) 

**FCN-8 - Encoder**

The encoder for FCN-8 is the VGG16 model pretrained on ImageNet for classification. The fully-connected layers of the VGG16 model are replaced by 1-by-1 convolution layers.

A 1-by-1 convolution is a result of sweeping a 1-by-1 kernel over the input with a sliding window and performing an element-wise multiplication and summation.

<img src="/images/1x1Convolution.png" width="480">

This will preserve the spatial information as we are not flattening out the output from previous layers and assuming interaction among individual input nodes, as in the fully-connected layers.

The number of kernels of a 1-by-1 convolution layer is equivalent to the number of outputs in a fully-connected layer, which is the number of output classes we are classifying.

In the context of this project, we are going to classify the image pixels into two classes, namely road or background. Thus, we will have two kernels for each 1-by-1 convolution layer that is applied to the VGG16 encoder layers.

**FCN-8 - Decoder**

To build the decoder portion of FCN-8, we’ll upsample the input to the original image size using transposed convolutions. The shape of the tensor after the final convolutional transpose layer will be 4-dimensional: (batch_size, original_height, original_width, num_classes).

Transposed convolutions help in upsampling the previous layer to a higher resolution or dimension. Upsampling is a classic signal processing technique which is often accompanied by interpolation. 

We can use a transposed convolution to transfer patches of data onto a sparse matrix, then we can fill the sparse area of the matrix based on the transferred information. A helpful animations of transposed convolution is shown below:

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="images/conv_math_transposed/no_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="images/conv_math_transposed/arbitrary_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="images/conv_math_transposed/same_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="images/conv_math_transposed/full_padding_no_strides_transposed.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides, transposed</td>
    <td>Arbitrary padding, no strides, transposed</td>
    <td>Half padding, no strides, transposed</td>
    <td>Full padding, no strides, transposed</td>
  </tr>
  <tr>
    <td><img width="150px" src="images/conv_math_transposed/no_padding_strides_transposed.gif"></td>
    <td><img width="150px" src="images/conv_math_transposed/padding_strides_transposed.gif"></td>
    <td><img width="150px" src="images/conv_math_transposed/padding_strides_odd_transposed.gif"></td>
    <td></td>
  </tr>
  <tr>
    <td>No padding, strides, transposed</td>
    <td>Padding, strides, transposed</td>
    <td>Padding, strides, transposed (odd)</td>
    <td></td>
  </tr>
</table>

_Note: Blue maps are inputs, and cyan maps are outputs._

A complete list of convolution arithmetic animations can be found [here](https://github.com/vdumoulin/conv_arithmetic)

**Skip Connections**

One effect of convolutions or encoding in general is you narrow down the scope by looking closely at some picture and lose the bigger picture as a result.

So even if we were to decode the output of the encoder back to the original image image, some information has been lost.

Using skip connections is a way of retaining this information, by connecting the output of one layer to a non-adjacent layer.

In the context of this project, 1-by-1 convolution is applied to the pooling layers from the encoder and the output are combined with the up-scaled decoder layers (of the same size) using the element-wise addition operation.

<img src="/images/SkipConnection.png" width="640">

These skip connections allow the network to use information from multiple resolutions.  As a result, the network is able to make more precise segmentation decisions.

---
### Implementation Summary

The final model implementation is as illustrated below:

<img src="/images/SemanticSegmentation.png" width="840">

The parameters used in the training of the model is as summarized below:

|Parameter    | Value   |
|------------:|--------:|
|Epoch        | 12      |
|Batch size   | 10      |
|Keep prob    | 0.3     |
|Learning rate| 0.001   |

---
### Result

The training loss over 12 epoch is as shown below:

<img src="/images/loss.png" width="640">

---
### Installation

**GPU**
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.

**Install Anaconda:**

Follow the instructions on the [Anaconda download site](https://www.continuum.io/downloads).

**Create environment:**

Run the following in the project top directory
```
$ conda env create -f environment-gpu.yml
```

**Uninstall environment:**

To uninstall the environment:

```
$ conda env remove -n semantic-segmentation-gpu
```

**Activate environment:**

In order to use the environment, you will need to activate it. This must be done **each** time you open a new terminal window. 

```
$ conda activate semantic-segmentation-gpu
```

To exit the environment, simply close the terminal window or run the following command:

```
$ conda deactivate semantic-segmentation-gpu
```


**Dataset**
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

---
### How to run

Run the following command to run the project:
```
python main.py
```
