# cnn-benchmarks

Benchmarks for popular convolutional neural network models on CPU and different GPUs, with and without cuDNN.

Some general conclusions from this benchmarking:

- **Pascal Titan X > GTX 1080**: Across all models, the Pascal Titan X is **1.31x to 1.43x** faster than the GTX 1080 and **1.47x to 1.60x** faster than the Maxwell Titan X. This is without a doubt the best card you can get for deep learning right now.
- **GTX 1080 > Maxwell Titan X**: Across all models, the GTX 1080 is **1.10x to 1.15x** faster than the Maxwell Titan X.
- **ResNet > VGG**: ResNet-50 is faster than VGG-16 and more accurate than VGG-19 (7.02 vs 9.0); ResNet-101 is about the same speed as VGG-19 but much more accurate than VGG-16 (6.21 vs 9.0).
- **Always use cuDNN**: On the Pascal Titan X, cuDNN is **2.2x to 3.0x** faster than nn; on the GTX 1080, cuDNN is **2.0x to 2.8x** faster than nn; on the Maxwell Titan X, cuDNN is **2.2x to 3.0x** faster than nn.
- **GPUs are critical**: The Pascal Titan X with cuDNN is **49x to 74x** faster than dual Xeon E5-2630 v3 CPUs.

All benchmarks were run in Torch. 
The GTX 1080 and Maxwell Titan X benchmarks were run on a machine with dual
Intel Xeon E5-2630 v3 processors (8 cores each plus hyperthreading means 32
threads) and 64GB RAM running Ubuntu 14.04 with the CUDA 8.0 Release Candidate.
The Pascal Titan X benchmarks were run on a machine with an Intel Core i5-6500
CPU and 16GB RAM running Ubuntu 16.04 with the CUDA 8.0 Release Candidate.
The GTX 1080 Ti benchmarks were run on a machine with an Intel Core i7-7700 CPU
and 64GB RAM running Ubuntu 16.04 with the CUDA 8.0 release.

We benchmark all models with a minibatch size of 16 and an image size of 224 x 224;
this allows direct comparisons between models, and allows all but the ResNet-200 model
to run on the GTX 1080, which has only 8GB of memory.

The following models are benchmarked:

|Network|Layers|Top-1 error|Top-5 error|Speed (ms)|Citation|
|---|---:|---:|---:|---:|---|
|[AlexNet](#alexnet)|8|42.90|19.80|14.56|[[1]](#alexnet-paper)|
|[Inception-V1](#inception-v1)|22|-|10.07|39.14|[[2]](#inception-v1-paper)|
|[VGG-16](#vgg-16)|16|27.00|8.80|128.62|[[3]](#vgg-paper)|
|[VGG-19](#vgg-19)|19|27.30|9.00|147.32|[[3]](#vgg-paper)|
|[ResNet-18](#resnet-18)|18|30.43|10.76|31.54|[[4]](#resnet-cvpr)|
|[ResNet-34](#resnet-34)|34|26.73|8.74|51.59|[[4]](#resnet-cvpr)|
|[ResNet-50](#resnet-50)|50|24.01|7.02|103.58|[[4]](#resnet-cvpr)|
|[ResNet-101](#resnet-101)|101|22.44|6.21|156.44|[[4]](#resnet-cvpr)|
|[ResNet-152](#resnet-152)|152|22.16|6.16|217.91|[[4]](#resnet-cvpr)|
|[ResNet-200](#resnet-200)|200|21.66|5.79|296.51|[[5]](#resnet-eccv)|

Top-1 and Top-5 error are single-crop error rates on the ILSVRC 2012 Validation set,
except for VGG-16 and VGG-19 which instead use dense prediction on a 256x256 image.
This gives the VGG models a slight advantage, but I was unable to find single-crop error
rates for these models. All models perform better when using more than one crop at test-time.

Speed is the total time for a forward and backward pass on a Pascal Titan X with cuDNN 5.1.

You can download the model files used for benchmarking [here](https://drive.google.com/open?id=0Byvt-AfX75o1STUxZTFpMU10djA) (2.1 GB);
these were converted from Caffe or Torch checkpoints using the `convert_model.lua` script.

We use the following GPUs for benchmarking:

|GPU|Memory|Architecture|CUDA Cores|FP32 TFLOPS|Release Date|
|---|---|---|---:|---:|---|
|[Pascal Titan X](http://www.geforce.com/hardware/10series/titan-x-pascal)|12GB GDDRX5|Pascal|3584|10.16|August 2016|
|[GTX 1080](http://www.geforce.com/hardware/10series/geforce-gtx-1080)|8GB GDDRX5|Pascal|2560|8.87|May 2016|
|[GTX 1080 Ti](https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1080-ti/)|11GB GDDRX5|Pascal|3584|10.6|March 2017|
|[Maxwell Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x)|12GB GDDR5|Maxwell|3072|6.14|March 2015|


## AlexNet
(input 16 x 3 x 224 x 224)

We use the [BVLC AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) from Caffe.

AlexNet uses grouped convolutions; this was a strategy to allow model parallelism over two GTX 580
GPUs, which had only 3GB of memory each. Grouped convolutions are no longer commonly used, and are
not even implemented by the [torch/nn](https://github.com/torch/nn) backend; therefore we can only
benchmark AlexNet using cuDNN.

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|GTX 1080 Ti|5.1.10|4.31|9.58|13.89|
|Pascal Titan X|5.1.05|5.04|9.52|14.56|
|Pascal Titan X|5.0.05|5.32|10.90|16.23|
|GTX 1080|5.1.05|7.00|13.74|20.74|
|Maxwell Titan X|5.1.05|7.09|14.76|21.85|
|GTX 1080|5.0.05|7.35|15.73|23.08|
|Maxwell Titan X|5.0.05|7.55|17.78|25.33|
|Maxwell Titan X|4.0.07|8.03|17.91|25.94|


## Inception-V1
(input 16 x 3 x 224 x 224)

We use the Torch implementation of Inception-V1 from
[soumith/inception.torch](https://github.com/soumith/inception.torch).

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|GTX 1080 Ti|5.1.10|11.50|25.37|36.87|
|Pascal Titan X|5.1.05|12.06|27.08|39.14|
|Pascal Titan X|5.0.05|11.94|28.39|40.33|
|GTX 1080|5.0.05|16.08|40.08|56.16|
|Maxwell Titan X|5.1.05|19.29|42.69|61.98|
|Maxwell Titan X|5.0.05|19.27|46.41|65.68|
|Maxwell Titan X|4.0.07|21.04|49.41|70.45|
|GTX 1080 Ti|None|56.34|85.30|141.64|
|Pascal Titan X|None|57.46|85.90|143.36|
|GTX 1080|None|63.03|102.31|165.34|
|Maxwell Titan X|None|91.31|140.81|232.12|


## VGG-16
(input 16 x 3 x 224 x 224)

This is Model D in [[3]](#vgg-paper) used in the ILSVRC-2014 competition,
[available here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md).

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|GTX 1080 Ti|5.1.10|41.23|86.91|128.14|
|Pascal Titan X|5.1.05|41.59|87.03|128.62|
|Pascal Titan X|5.0.05|46.16|111.23|157.39|
|GTX 1080|5.1.05|59.37|123.42|182.79|
|Maxwell Titan X|5.1.05|62.30|130.48|192.78|
|GTX 1080|5.0.05|67.27|166.17|233.43|
|Maxwell Titan X|5.0.05|75.80|186.47|262.27|
|Maxwell Titan X|4.0.07|111.99|226.69|338.69|
|Pascal Titan X|None|98.15|260.38|358.53|
|GTX 1080|None|143.73|379.09|522.82|
|Maxwell Titan X|None|172.61|415.87|588.47|
|CPU: Dual Xeon E5-2630 v3|None|3101.76|5393.72|8495.48|



## VGG-19
(input 16 x 3 x 224 x 224)

This is Model E in [[3]](#vgg-paper) used in the ILSVRC-2014 competition,
[available here](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md).


|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|Pascal Titan X|5.1.05|48.09|99.23|147.32|
|GTX 1080 Ti|5.1.10|48.15|100.04|148.19|
|Pascal Titan X|5.0.05|55.75|134.98|190.73|
|GTX 1080|5.1.05|68.95|141.44|210.39|
|Maxwell Titan X|5.1.05|73.66|151.48|225.14|
|GTX 1080|5.0.05|79.79|202.02|281.81|
|Maxwell Titan X|5.0.05|93.47|229.34|322.81|
|Maxwell Titan X|4.0.07|139.01|279.21|418.22|
|Pascal Titan X|None|121.69|318.39|440.08|
|GTX 1080|None|176.36|453.22|629.57|
|Maxwell Titan X|None|215.92|491.21|707.13|
|CPU: Dual Xeon E5-2630 v3|None|3609.78|6239.45|9849.23|



## ResNet-18
(input 16 x 3 x 224 x 224)

This is the 18-layer model described in [[4]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|Pascal Titan X|5.1.05|10.14|21.40|31.54|
|GTX 1080 Ti|5.1.10|10.45|22.34|32.78|
|Pascal Titan X|5.0.05|10.06|23.08|33.13|
|GTX 1080|5.1.05|14.62|29.32|43.94|
|GTX 1080|5.0.05|14.84|32.68|47.52|
|Maxwell Titan X|5.1.05|16.87|34.55|51.42|
|Maxwell Titan X|5.0.05|17.08|37.79|54.87|
|Maxwell Titan X|4.0.07|21.54|42.26|63.80|
|Pascal Titan X|None|34.76|61.64|96.40|
|GTX 1080 Ti|None|50.04|65.99|116.03|
|GTX 1080|None|42.94|79.17|122.10|
|Maxwell Titan X|None|55.82|96.01|151.82|
|CPU: Dual Xeon E5-2630 v3|None|847.46|1348.33|2195.78|



## ResNet-34
(input 16 x 3 x 224 x 224)

This is the 34-layer model described in [[4]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|GTX 1080 Ti|5.1.10|16.71|34.60|51.31|
|Pascal Titan X|5.1.05|17.01|34.58|51.59|
|Pascal Titan X|5.0.05|16.91|38.67|55.58|
|GTX 1080|5.1.05|24.50|47.59|72.09|
|GTX 1080|5.0.05|24.76|55.00|79.76|
|Maxwell Titan X|5.1.05|27.33|52.90|80.23|
|Maxwell Titan X|5.0.05|28.79|63.19|91.98|
|Maxwell Titan X|4.0.07|40.12|76.00|116.11|
|Pascal Titan X|None|66.56|106.42|172.98|
|GTX 1080 Ti|None|86.30|109.43|195.73|
|GTX 1080|None|82.71|137.42|220.13|
|Maxwell Titan X|None|108.95|166.19|275.13|
|CPU: Dual Xeon E5-2630 v3|None|1530.01|2435.20|3965.21|


## ResNet-50
(input 16 x 3 x 224 x 224)

This is the 50-layer model described in [[4]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|GTX 1080 Ti|5.1.10|34.14|67.06|101.21|
|Pascal Titan X|5.1.05|35.03|68.54|103.58|
|Pascal Titan X|5.0.05|35.03|70.76|105.78|
|GTX 1080|5.1.05|50.64|99.18|149.82|
|GTX 1080|5.0.05|50.76|103.35|154.11|
|Maxwell Titan X|5.1.05|55.75|103.87|159.62|
|Maxwell Titan X|5.0.05|56.30|109.75|166.05|
|Maxwell Titan X|4.0.07|62.03|116.81|178.84|
|Pascal Titan X|None|87.62|158.96|246.58|
|GTX 1080 Ti|None|99.90|177.58|277.47|
|GTX 1080|None|109.79|201.40|311.18|
|Maxwell Titan X|None|137.14|247.65|384.79|
|CPU: Dual Xeon E5-2630 v3|None|2477.61|4149.64|6627.25|



## ResNet-101
(input 16 x 3 x 224 x 224)

This is the 101-layer model described in [[4]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|GTX 1080 Ti|5.1.10|52.18|102.08|154.26|
|Pascal Titan X|5.1.05|53.38|103.06|156.44|
|Pascal Titan X|5.0.05|53.28|108.20|161.48|
|GTX 1080|5.1.05|77.59|148.21|225.80|
|GTX 1080|5.0.05|77.39|158.19|235.58|
|Maxwell Titan X|5.1.05|87.76|159.73|247.49|
|Maxwell Titan X|5.0.05|88.45|172.12|260.57|
|Maxwell Titan X|4.0.07|108.96|189.93|298.90|
|Pascal Titan X|None|161.55|257.57|419.11|
|GTX 1080 Ti|None|162.03|266.77|428.81|
|GTX 1080|None|203.19|322.48|525.67|
|Maxwell Titan X|None|260.48|453.45|713.93|
|CPU: Dual Xeon E5-2630 v3|None|4414.91|6891.33|11306.24|



## ResNet-152
(input 16 x 3 x 224 x 224)

This is the 152-layer model described in [[4]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|GTX 1080 Ti|5.1.10|73.52|142.02|215.54|
|Pascal Titan X|5.1.05|75.45|142.47|217.91|
|Pascal Titan X|5.0.05|75.12|150.08|225.20|
|GTX 1080|5.1.05|109.32|204.98|314.30|
|GTX 1080|5.0.05|109.64|218.62|328.26|
|Maxwell Titan X|5.1.05|124.04|221.41|345.45|
|Maxwell Titan X|5.0.05|124.88|240.16|365.03|
|Maxwell Titan X|4.0.07|150.90|268.64|419.54|
|Pascal Titan X|None|238.04|371.40|609.43|
|GTX 1080 Ti|None|225.36|368.42|593.79|
|GTX 1080|None|299.05|461.67|760.72|
|Maxwell Titan X|None|382.39|583.83|966.22|
|CPU: Dual Xeon E5-2630 v3|None|6572.17|10300.61|16872.78|


## ResNet-200
(input 16 x 3 x 224 x 224)

This is the 200-layer model described in [[5]](#resnet-eccv) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

Even with a batch size of 16, the 8GB GTX 1080 did not have enough memory to run
the model.

|GPU|cuDNN|Forward (ms)|Backward (ms)|Total (ms)|
|---|---|---:|---:|---:|
|Pascal Titan X|5.1.05|104.74|191.77|296.51|
|Pascal Titan X|5.0.05|104.36|201.92|306.27|
|Maxwell Titan X|5.0.05|170.03|320.80|490.83|
|Maxwell Titan X|5.1.05|169.62|383.80|553.42|
|Maxwell Titan X|4.0.07|203.52|356.35|559.87|
|Pascal Titan X|None|314.77|519.72|834.48|
|Maxwell Titan X|None|497.57|953.94|1451.51|
|CPU: Dual Xeon E5-2630 v3|None|8666.43|13758.73|22425.16|

## Citations

<a id='alexnet-paper'>
[1] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet Classification with Deep Convolutional Neural Networks." NIPS 2012
<br>

<a id='inception-v1-paper'>
[2] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
Dragomir Anguelov, Dumitru Erhan, Andrew Rabinovich.
"Going Deeper with Convolutions." CVPR 2015.
<br>

<a id='vgg-paper'>
[3] Karen Simonyan and Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition." ICLR 2015
<br>

<a id='resnet-cvpr'>
[4] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep Residual Learning for Image Recognition." CVPR 2016.
<br>

<a id='resnet-eccv'>
[5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Identity Mappings in Deep Residual Networks." ECCV 2016.
