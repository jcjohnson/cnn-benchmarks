# cnn-benchmarks

Benchmarks for popular convolutional neural network models on CPU and different GPUs, with and without cuDNN.

Some general conclusions from this benchmarking:

- **Pascal Titan X > GTX 1080**: Across all models, the Pascal Titan X is **1.31x to 1.43x** faster than the GTX 1080 and **1.47x to 1.60x** faster than the Maxwell Titan X.
- **GTX 1080 > Maxwell Titan X**: Across all models, the GTX 1080 is **1.10x to 1.15x** faster than the Maxwell Titan X.
- **ResNet > VGG**: ResNet-50 is **1.5x** faster than VGG-16 and more accurate than VGG-19 (7.02 vs 8.0); ResNet-101 is about the same speed as VGG-16 but much more accurate than VGG-19 (6.21 vs 8.0).
- **Always use cuDNN**: On the GTX 1080, cuDNN is **2.0x to 2.8x** faster than nn; on the Maxwell Titan X, cuDNN is **2.2x to 3.0x** faster than nn.
- **GPUs are critical**: The GTX 1080 with cuDNN is **35x to 50x** faster than dual Xeon E5-2630 v3 CPUs.

All benchmarks were run in Torch. 
The GTX 1080 and Maxwell Titan X benchmarks were run on a machine with dual
Intel Xeon E5-2630 v3 processors (8 cores each plus hyperthreading means 32
threads) and 64GB RAM running Ubuntu 14.04 with the CUDA 8.0 Release Candidate.
The Pascal Titan X benchmarks were run on a machine with an Intel Core i5-6500
CPU and 16GB RAM running Ubuntu 16.04 with the CUDA 8.0 Release Candidate.

We benchmark all models with a minibatch size of 16 and an image size of 224 x 224;
this allows direct comparisons between models, and allows all but the ResNet-200 model
to run on the GTX 1080, which has only 8GB of memory.

The following models are benchmarked:

|Network|Layers|Top-1 error|Top-5 error|Speed (ms)|Citation|
|---|---:|---:|---:|---:|---|
|[AlexNet](#alexnet)|8|42.90|19.80|16.12|[[1]](#alexnet-paper)|
|[VGG-16](#vgg-16)|16|25.60|8.10|165.00|[[2]](#vgg-paper)|
|[VGG-19](#vgg-19)|19|25.50|8.00|201.82|[[2]](#vgg-paper)|
|[ResNet-18](#resnet-18)|18|30.43|10.76|33.47|[[3]](#resnet-cvpr)|
|[ResNet-34](#resnet-34)|34|26.73|8.74|57.45|[[3]](#resnet-cvpr)|
|[ResNet-50](#resnet-50)|50|24.01|7.02|110.18|[[3]](#resnet-cvpr)|
|[ResNet-101](#resnet-101)|101|22.44|6.21|167.61|[[3]](#resnet-cvpr)|
|[ResNet-152](#resnet-152)|152|22.16|6.16|229.49|[[3]](#resnet-cvpr)|
|[ResNet-200](#resnet-200)|200|21.66|5.79|310.80|[[4]](#resnet-eccv)|

Top-1 and Top-5 error are single-crop error rates on the ILSVRC 2012 Validation set.
Speed is the total time for a forward and backward pass on a Pascal Titan X with cuDNN 5.0.

We use the following GPUs for benchmarking:

|GPU|Memory|Architecture|CUDA Cores|FP32 TFLOPS|Release Date|
|---|---|---|---:|---:|---|
|GeForce GTX Titan X|12GB GDDR5|Maxwell|3072|6.14|March 2015|
|GeForce GTX 1080|8GB GDDRX5|Pascal|2560|8.87|May 2016|
|TITAN X|12GB GDDRX5|Pascal|3584|10.16|August 2016|


## AlexNet
(input 16 x 3 x 224 x 224)

We use the [BVLC AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) from Caffe.

AlexNet uses grouped convolutions; this was a strategy to allow model parallelism over two GTX 580
GPUs, which had only 3GB of memory each. Grouped convolutions are no longer commonly used, and are
not even implemented by the [torch/nn](https://github.com/torch/nn) backend; therefore we can only
benchmark AlexNet using cuDNN.

|GPU|Forward (ms)|Backward (ms)|Total (ms)|
|---|---:|---:|---:|
|TITAN X (cuDNN 5005)|5.29|10.83|16.12|
|GeForce GTX 1080 (cuDNN 5005)|7.36|15.83|23.18|
|GeForce GTX TITAN X (cuDNN 5005)|7.02|16.69|23.71|


## VGG-16
(input 16 x 3 x 224 x 224)

This is Model D in [[2]](#vgg-paper) used in the ILSVRC-2014 competition,
[available here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md).

|GPU|Forward (ms)|Backward (ms)|Total (ms)|
|---|---:|---:|---:|
|TITAN X (cuDNN 5005)|46.01|119.00|165.00|
|GeForce GTX 1080 (cuDNN 5005)|66.56|165.98|232.55|
|GeForce GTX TITAN X (cuDNN 5005)|76.15|186.28|262.42|
|TITAN X (nn)|102.49|269.81|372.30|
|GeForce GTX 1080 (nn)|143.81|378.61|522.42|
|GeForce GTX TITAN X (nn)|172.56|415.41|587.97|
|CPU: Dual Intel Xeon E5-2630 v3|3101.76|5393.72|8495.48|


## VGG-19
(input 16 x 3 x 224 x 224)

This is Model E in [[2]](#vgg-paper) used in the ILSVRC-2014 competition,
[available here](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md).

|GPU|Forward (ms)|Backward (ms)|Total (ms)|
|---|---:|---:|---:|
|TITAN X (cuDNN 5005)|55.88|145.94|201.82|
|GeForce GTX 1080 (cuDNN 5005)|80.39|201.31|281.69|
|GeForce GTX TITAN X (cuDNN 5005)|93.83|229.57|323.40|
|TITAN X (nn)|126.86|318.54|445.40|
|GeForce GTX 1080 (nn)|176.45|453.63|630.08|
|GeForce GTX TITAN X (nn)|215.55|494.33|709.88|
|CPU: Dual Intel Xeon E5-2630 v3|3609.78|6239.45|9849.23|


## ResNet-18
(input 16 x 3 x 224 x 224)

This is the 18-layer model described in [[3]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|Forward (ms)|Backward (ms)|Total (ms)|
|---|---:|---:|---:|
|TITAN X (cuDNN 5005)|10.14|23.33|33.47|
|GeForce GTX 1080 (cuDNN 5005)|14.69|32.38|47.07|
|GeForce GTX TITAN X (cuDNN 5005)|16.97|36.86|53.84|
|TITAN X (nn)|34.55|62.39|96.94|
|GeForce GTX 1080 (nn)|43.05|78.95|122.00|
|GeForce GTX TITAN X (nn)|55.20|95.57|150.76|
|CPU: Dual Intel Xeon E5-2630 v3|847.46|1348.33|2195.78| 


## ResNet-34
(input 16 x 3 x 224 x 224)

This is the 34-layer model described in [[3]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|Forward (ms)|Backward (ms)|Total (ms)|
|---|---:|---:|---:|
|TITAN X (cuDNN 5005)|17.36|40.09|57.45|
|GeForce GTX 1080 (cuDNN 5005)|24.83|54.86|79.70|
|GeForce GTX TITAN X (cuDNN 5005)|28.72|63.22|91.94|
|TITAN X (nn)|66.46|107.99|174.45|
|GeForce GTX 1080 (nn)|84.27|138.04|222.31|
|GeForce GTX TITAN X (nn)|109.75|164.94|274.69|
|CPU: Dual Intel Xeon E5-2630 v3|1530.01|2435.20|3965.21|


## ResNet-50
(input 16 x 3 x 224 x 224)

This is the 50-layer model described in [[3]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|Forward (ms)|Backward (ms)|Total (ms)|
|---|---:|---:|---:|
|TITAN X (cuDNN 5005)|36.25|73.93|110.18|
|GeForce GTX 1080 (cuDNN 5005)|50.67|103.24|153.90|
|GeForce GTX TITAN X (cuDNN 5005)|56.42|114.60|171.02|
|TITAN X (nn)|87.81|161.03|248.83|
|GeForce GTX 1080 (nn)|109.81|201.66|311.47|
|GeForce GTX TITAN X (nn)|136.37|245.99|382.36|
|CPU: Dual Intel Xeon E5-2630 v3|2477.61|4149.64|6627.25|


## ResNet-101
(input 16 x 3 x 224 x 224)

This is the 101-layer model described in [[3]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|Forward (ms)|Backward (ms)|Total (ms)|
|---|---:|---:|---:|
|TITAN X (cuDNN 5005)|54.48|113.14|167.61|
|GeForce GTX 1080 (cuDNN 5005)|77.77|157.56|235.33|
|GeForce GTX TITAN X (cuDNN 5005)|88.30|171.82|260.12|
|TITAN X (nn)|165.37|262.78|428.15|
|GeForce GTX 1080 (nn)|203.33|321.60|524.93|
|GeForce GTX TITAN X (nn)|258.26|404.16|662.42|
|CPU: Dual Intel Xeon E5-2630 v3|4414.91|6891.33|11306.24|


## ResNet-152
(input 16 x 3 x 224 x 224)

This is the 101-layer model described in [[3]](#resnet-cvpr) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

|GPU|Forward (ms)|Backward (ms)|Total (ms)|
|---|---:|---:|---:|
|TITAN X (cuDNN 5005)|75.87|153.62|229.49|
|GeForce GTX 1080 (cuDNN 5005)|109.93|218.97|328.90|
|GeForce GTX TITAN X (cuDNN 5005)|125.69|241.28|366.97|
|TITAN X (nn)|250.93|390.46|641.40|
|GeForce GTX 1080 (nn)|299.12|460.95|760.07|
|GeForce GTX TITAN X (nn)|379.79|579.63|959.42|
|CPU: Dual Intel Xeon E5-2630 v3|6572.17|10300.61|16872.78|

## ResNet-200
(input 16 x 3 x 224 x 224)

This is the 200-layer model described in [[4]](#resnet-eccv) and implemented in 
[fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

Even with a batch size of 16, the 8GB GTX 1080 did not have enough memory to run
the model.

|GPU|Forward (ms)|Backward (ms)|Total (ms)|
|---|---:|---:|---:|
|TITAN X (cuDNN 5005)|104.98|205.82|310.80|
|GeForce GTX TITAN X (cuDNN 5005)|171.15|322.66|493.82|
|TITAN X (nn)|313.90|522.16|836.05|
|GeForce GTX TITAN X (nn)|491.69|806.95|1298.65|
|CPU: Dual Intel Xeon E5-2630 v3|8666.43|13758.73|22425.16|

## Citations

<a id='alexnet-paper'>
[1] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet Classification with Deep Convolutional Neural Networks." NIPS 2012

<a id='vgg-paper'>
[2] Karen Simonyan and Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition." ICLR 2015

<a id='resnet-cvpr'>
[3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep Residual Learning for Image Recognition." CVPR 2016.

<a id='resnet-eccv'>
[4] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Identity Mappings in Deep Residual Networks." ECCV 2016.
