## Overview
Here includes some of my experiments and implementation of some technics from lecture 8 , they are:

- :heavy_check_mark: optical flow in OpenCV.(DO NOT USE THE `visualize_optical_flow` in `optical-flow.py`)

- :heavy_check_mark: histogram and histogram equalization.


## Usage 

- optical flow

modify the video file or just feed in nothing to use your camera, and run:
```
$ python optical-flow.py
```
- histogram equalization

modify the image file of `Line-144` of `histogram.py` and run:
```
$ python histogram.py
```


## My Result
- Result of Optical-Flow

Input: `/video/car_moving.mp4 `

Output:

![](./result/frame.png) 

![](./result/optical-hsv.png)
 
 - Result of Historgram Equalization

Input: 

![](./img/x-ray.jpg)

Output:

![](./result/hist-equ.jpg) 
