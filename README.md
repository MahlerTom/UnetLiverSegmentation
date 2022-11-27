# LiverSegmentation Project

In this project we implemented a U-Net convolutional neural network in `Tensorflow 2.0` and `keras`, inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ "U-Net: Convolutional Networks for Biomedical Image Segmentation"). Our implementation is based on [zhixuhao/unet](https://github.com/zhixuhao/unet "zhixuhao/unet"). Our implementation is contained within the [LiverSegmentationReport Jupyter Notebook](../blob/master/LiverSegmentationReport.ipynb "LiverSegmentationReport").

## Introduction

A fully automatic technique for segmenting the liver and localizing its unhealthy tissues is essential in many clinical applications, such as pathological diagnosis, surgical planning, and postoperative assessment. However, it is still a very challenging task due to the complex background, fuzzy boundary, and various appearance of both liver and liver lesions.

### Tasks
In this project we will develop automatic algorithms to segment liver and liver lesions in abdominal CT scans. To achieve this goal we'll utilize the power of deep learning algorithms.

The project consists of the following tasks:

1.  Liver segmentation
2.  Lesions segmentation

### Data

The training data set consists of 11 CT scans:

* `Data` directory contains CT images converted to `png` format
* `Segmentation` directory contains segmentation masks:
  * Pixels with value 127 indicate liver
  * Pixels with value 255 indicate liver lesion

The data can be downloaded from the following link: [data link](https://drive.google.com/open?id=1lhYdOFymZSC5Gz76Zt4GzcDYc8nWaWJv).

### Evaluation metrics
The following metrics will be used to evaluate the segmentation accuracy:

* Dice similarity coeffcient (Dice): 
  
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;Dice=\frac{2\times&space;TP}{2\times&space;TP+FP+FN}" title="\Large Dice=\frac{2\times TP}{2\times TP+FP+FN}" />

  TP, FP, and FN denote the number of true positive, false positive, and false negative pixels respectively. Dice computes a normalized overlap value between the produced and ground truth segmentation.

* Precision (positive predictive rate):

  <img src="https://latex.codecogs.com/svg.latex?\Large&space;Precision=\frac{TP}{TP+FP}" title="\Large Precision=\frac{TP}{TP+FP}" />
  
  Precision expresses the proportion between the true segmented pixels and all pixels the model associates with liver / lesions.

* Recall (sensitivity):
  
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;Recall=\frac{TP}{TP+FN}" title="\Large Recall=\frac{TP}{TP+FN}" />
  
  Recall expresses the proportion between the true segmented pixels and all liver / lesions pixels.

## Getting Started

The [LiverSegmentationReport.ipynb](../blob/master/LiverSegmentationReport.ipynb "LiverSegmentationReport") is completely self contained when run on [Google Colab](https://colab.research.google.com/ "Welcome To Colaboratory - Colaboratory - Google") notebook for printing the results.

### Installing `TensorFlow 2.0`

We use of [`TensorFlow 2.0`](https://www.tensorflow.org/alpha) API on [Google Colab](https://colab.research.google.com/ "Welcome To Colaboratory - Colaboratory - Google") notebook set on GPU runtime (GPU: 1xTesla K80 , having 2496 CUDA cores, compute 3.7, 12GB (11.439GB Usable) GDDR5 VRAM).

We load `TF2.0` as follows:
```
pip install -q tensorflow-gpu==2.0.0-alpha0
```

Note that you should make sure to use the GPU runtime, you can check it by running the following command inside the notebook:
```python
print('Running on GPU' if tf.test.is_gpu_available() else 'Please change runtime type to GPU on Google Colab under Runtime') # Make sure we are set to GPU (under Runtime->Change runtime type)
```

# How to run the notebook

Open the notebook in google colab and run all cells. The notebook presents our experiments results. The training and validation images are placed in data\train and data\val directories. In addition, we add the test data in data\test. In order to test the code with different images, please replace the images inside data\test with your own images.

To use the notebook with costume configurations, please run all cells until the Results section. Then, you may add new code cells and use the functions as follows:
