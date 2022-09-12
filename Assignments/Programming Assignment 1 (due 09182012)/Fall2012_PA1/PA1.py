# -*- coding: utf-8 -*-
"""
    @Author  : LiuZhian
    @Time    : 2019/8/5 0005 下午 4:06
    @Comment : 
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_kernel_generator(sigma):
	"""
	This function generate a 2D Gaussian Kernel, which follows the
	below given Gaussian Distribution:

	G(x,y)=1/(2*pi*sigma*sigma)*exp(-(x**2+y**2)/2/sigma**2)

	Where, y is the distance along vertical axis from the origin,
	x is the distance along horizontal axis from the origin
	and σ is the standard deviation.

	:param sigma: standard deviation.
	:return:  a 2D Gaussian Kernel with (2*sigma+1)x(2*sigma+1) as size
	"""
	# we assume sigma is a positive integer
	assert (type(sigma) == int and sigma > 0)
	s = 2 * sigma * sigma
	# sum for normalization
	sum_n = 0.0

	GKernel = np.zeros((2 * sigma + 1, 2 * sigma + 1), dtype=np.float32)

	for x in range(-sigma, sigma + 1):
		for y in range(-sigma, sigma + 1):
			r = x * x + y * y
			GKernel[x + sigma, y + sigma] = np.exp(-r / s) / np.sqrt(np.pi * s)
			sum_n += GKernel[x + sigma, y + sigma]

	# normalize the kernel
	GKernel /= sum_n

	return GKernel


def convolution(img, k_type, k_size, sigma=None):
	kernel = None
	if k_type == "avg":
		kernel = np.ones((k_size, k_size), dtype=np.float32) / k_size / k_size
	elif k_type == "gaussian":
		kernel = gaussian_kernel_generator(sigma)
	else:
		raise Exception("the k_type currently only support avg or gaussian ")

	dst = cv2.filter2D(img, -1, kernel)

	return dst


def conv_test():
	noise_img = cv2.imread("balloonGrayNoisy.jpg")

	avg_3_denoising = convolution(noise_img, "avg", 3)
	avg_5_denoising = convolution(noise_img, "avg", 5)
	avg_7_denoising = convolution(noise_img, "avg", 7)

	gaussian_3_denoising = convolution(noise_img, "gaussian", 3, sigma=1)
	gaussian_5_denoising = convolution(noise_img, "gaussian", 5, sigma=2)
	gaussian_7_denoising = convolution(noise_img, "gaussian", 7, sigma=3)

	plt.subplot(231)
	plt.title("avg size-3"), plt.xticks([]), plt.yticks([])
	plt.imshow(avg_3_denoising[:, :, ::-1])
	plt.subplot(232)
	plt.title("avg size-5"), plt.xticks([]), plt.yticks([])
	plt.imshow(avg_5_denoising[:, :, ::-1])
	plt.subplot(233)
	plt.title("avg size-7"), plt.xticks([]), plt.yticks([])
	plt.imshow(avg_7_denoising[:, :, ::-1])

	plt.subplot(234)
	plt.title("gaussian size-3"), plt.xticks([]), plt.yticks([])
	plt.imshow(gaussian_3_denoising[:, :, ::-1])
	plt.subplot(235)
	plt.title("gaussian size-5"), plt.xticks([]), plt.yticks([])
	plt.imshow(gaussian_5_denoising[:, :, ::-1])
	plt.subplot(236)
	plt.title("gaussian size-7"), plt.xticks([]), plt.yticks([])
	plt.imshow(gaussian_7_denoising[:, :, ::-1])

	plt.show()
	# plt.savefig("result.png")


def edge_detect(img, k_type, threshold):
	"""
	Edge detection using sobel or prewitt operator.

	The main processing obeys the following steps:
	0. generate the operator
	1. noise removing (using gaussian smoothing)
	2. compute horizontal and vertical gradients
	3. magnitude the gradients
	4. apply a threshold

	:param img: input image
	:param k_type: kernel type(sobel or prewitt)
	:return:
	"""
	# step 0: generate the operator
	if k_type == "sobel":
		kernel_x = np.zeros((3, 3), dtype=np.float32)
		kernel_y = np.zeros((3, 3), dtype=np.float32)
		kernel_x[0, :] = [-1, -2, -1]
		kernel_x[2, :] = [1, 2, 1]
		kernel_y[:, 0] = [-1, -2, -1]
		kernel_y[:, 2] = [1, 2, 1]

	elif k_type == "prewitt":
		kernel_x = np.zeros((3, 3), dtype=np.float32)
		kernel_y = np.zeros((3, 3), dtype=np.float32)
		kernel_x[0, :] = [1, 1, 1]
		kernel_x[2, :] = [-1, -1, -1]
		kernel_y[:, 0] = [-1, -1, -1]
		kernel_y[:, 2] = [1, 1, 1]
	else:
		raise Exception("the k_type currently only support avg or gaussian ")

	# step 1: gaussian smoothing
	# **** you can try different kernel size here  ****
	img = cv2.GaussianBlur(img, (11, 11), 0)
	# step 2: compute horizontal and vertical gradients
	grad_x = cv2.filter2D(img, -1, kernel_x)
	grad_y = cv2.filter2D(img, -1, kernel_y)

	# step 3: magnitude the gradients
	magnitution = np.sqrt(grad_x ** 2 + grad_y ** 2)

	# step 4: apply a threshold
	edge_img = np.zeros_like(magnitution, dtype=np.uint8)
	edge_img[magnitution > threshold] = 255

	return edge_img

def edge_test():
	img = cv2.imread("buildingGray.jpg", cv2.IMREAD_GRAYSCALE)

	threshold = [10, 11, 12]
	plt.figure(figsize=(6, 4.8))
	for i in range(3):
		plt.subplot(2, 3, 1 + i)
		plt.title("sobel threshold-%s" % threshold[i]), plt.xticks([]), plt.yticks([])
		sobel_res = edge_detect(img, "sobel", threshold[i])
		plt.imshow(sobel_res, cmap="gray")

		plt.subplot(2, 3, 4 + i)
		plt.title("prewitt threshold-%s" % threshold[i]), plt.xticks([]), plt.yticks([])
		prewitt_res = edge_detect(img, "prewitt", threshold[i])
		plt.imshow(prewitt_res, cmap="gray")

	plt.show()
	# plt.savefig("edge-res.jpg")

if __name__ == '__main__':
	conv_test()
	edge_test()
