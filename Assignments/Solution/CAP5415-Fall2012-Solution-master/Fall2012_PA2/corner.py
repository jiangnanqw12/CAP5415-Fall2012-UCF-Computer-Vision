# -*- coding: utf-8 -*-
"""
    @Author  : LiuZhian
    @Time    : 2019/8/19 0019 上午 10:38
    @Comment : 
"""

import cv2
import sys

result = None


def on_trackbar_change(threshold):
	global result
	result = myHarris(img, 5, 3, 0.05, threshold)
	cv2.imshow('Harris-Detector', result)


def myHarris(src, blockSize, ksize, k, threshold):
	"""

	Harris Corner Detection Algorithm Steps:

	1. compute horiziontal and vertical derivatives Ix and Iy of the source image.
	   Specifically, we can utilise Sobel Operator to calculate them.
	2. compute three images corresponding to three terms in matrix M.
	3. convolve these three images with a larger Gaussian Kernel.
	   You can also specific a customized window.
	4. compute scalar cornerness response value using one of the R measures.
	5. find local maxima above some threshold as detected interest points.

	For better use and comparision with OpenCV built-in Harris implementation,
	I obey the similar in/out parameters of the API.

	:param src: 		Input image, it should be grayscale  type.
	:param blockSize: 	It is the size of neighbourhood considered for corner detection
	:param ksize: 		Aperture parameter of Sobel derivative used.
	:param k:  			Harris detector free parameter in the equation.
	:param threshold: 	Threshold for final choosing.
	:return:
	"""
	img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	# step 1: get first derivatives
	Ix = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=ksize)
	Iy = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=ksize)

	# step 2: get three terms in matrix M
	Ixx = Ix ** 2
	Ixy = Ix * Iy
	Iyy = Iy ** 2

	# step 3: Gaussian convolve respectively
	Sxx = cv2.GaussianBlur(Ixx, (blockSize, blockSize), 0)
	Sxy = cv2.GaussianBlur(Ixy, (blockSize, blockSize), 0)
	Syy = cv2.GaussianBlur(Iyy, (blockSize, blockSize), 0)

	# step 4: calculate R
	det = Sxx * Syy - Sxy ** 2
	trace = Sxx + Syy
	R = det - k * trace

	# step 5: apply threshold
	result = src.copy()
	result[R > threshold] = [0, 0, 255]

	return result


if __name__ == '__main__':
	image_name = sys.argv[1]
	img = cv2.imread(image_name)

	# Create a black image, a window
	cv2.namedWindow('Harris-Detector', cv2.WINDOW_AUTOSIZE)

	# create trackbars for threshold
	cv2.createTrackbar('threshold', 'Harris-Detector', 150, 255, on_trackbar_change)

	cv2.imshow('Harris-Detector', img)

	on_trackbar_change(cv2.getTrackbarPos('threshold', 'Harris-Detector'))

	if cv2.waitKey(0) == 32:
		cv2.imwrite("./result/%s-%s.jpg" % (image_name, cv2.getTrackbarPos('threshold', 'Harris-Detector')), result)
