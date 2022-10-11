# -*- coding: utf-8 -*-
"""
    @Author  : LiuZhian
    @Time    : 2019/8/16 0016 上午 10:57
    @Comment : 
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def LoG_kernel_generator(sigma, size):
	"""
	This function generate a 2D Laplacian of Gaussian kernel,
	which follows the bellow given formation:

	LoG(x,y)=[1/sqrt(2*pi*sigma**2)]*[(x**2+y**2-2*sigma*sigma)/(sigma**4)-2]*exp[-(x**2+y**2)/2/sigma**2]

	Where, y is the distance along vertical axis from the origin,
	x is the distance along horizontal axis from the origin
	and σ is the standard deviation in Gaussian kernel.

	:param sigma: standard deviation.
	:param n: kernel size
	:return:  a 2D Laplacian of Gaussian kernel with n*n as size
	"""
	# we assume sigma and n are a positive integers,and n must be an odd number
	assert (type(sigma) == int and sigma > 0)
	assert (type(size) == int and size > 0 and size % 2 == 1)

	x, y = np.meshgrid(np.arange(int(-size / 2), int(size / 2) + 1),
					   np.arange(int(-size / 2), int(size / 2) + 1))

	normal = 1 / (2.0 * np.pi * sigma ** 2)

	LoGKernel = ((x ** 2 + y ** 2 - (2.0 * sigma ** 2)) / sigma ** 4) * \
				np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2)) / normal

	# normalize the kernel

	sum_n = np.sum(LoGKernel)
	LoGKernel /= sum_n

	return LoGKernel


def Marr_Hidreth_detect(img_path, threshold_p=0.4, sigma=4, size=31):
	img_origin = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)

	# zero padding
	p = (size - 1) // 2
	img = np.zeros((2 * p + img_origin.shape[0], 2 * p + img_origin.shape[1]))
	img[p:-p, p:-p] = img_origin

	# generate LoG filter
	LoG_kernel = LoG_kernel_generator(sigma, size)

	# convolution operation
	img_LoG = np.zeros_like(img, dtype=float)

	# applying filter
	for i in range(img.shape[0] - size + 1):
		for j in range(img.shape[1] - size + 1):
			window = img[i:i + size, j:j + size] * LoG_kernel
			img_LoG[i, j] = np.sum(window)

	img_LoG = img_LoG.astype(np.int64, copy=False)

	# find maximum of the img_LoG, this is for choosing threshold conveniently.
	max_L = np.max(img_LoG)
	threshold = int(max_L * threshold_p)

	edge = np.zeros_like(img_LoG)

	# find zero-crossing pixel
	rows, cols = img_LoG.shape
	for r in range(rows):
		for c in range(cols):
			# search for opposite neighbors pairs
			up = r - 1
			down = r + 1
			left = c - 1
			right = c + 1
			# up-left and down-right pair
			if up >= 0 and left >= 0 and down < rows and right < cols:
				if img_LoG[up, left] * img_LoG[down, right] < 0 and abs(
						img_LoG[up, left] - img_LoG[down, right]) > threshold:
					edge[r, c] = 255
			# up and down pair
			if up >= 0 and down < rows:
				if img_LoG[up, c] * img_LoG[down, c] < 0 and abs(
						img_LoG[up, c] - img_LoG[down, c]) > threshold:
					edge[r, c] = 255
			# down-left and up-right pair
			if up >= 0 and left >= 0 and down < rows and right < cols:
				if img_LoG[down, left] * img_LoG[up, right] < 0 and abs(
						img_LoG[down, left] - img_LoG[up, right]) > threshold:
					edge[r, c] = 255
			# left and right pair
			if left >= 0 and right < cols:
				if img_LoG[r, left] * img_LoG[r, right] < 0 and abs(
						img_LoG[r, left] - img_LoG[r, right]) > threshold:
					edge[r, c] = 255
	return edge


def Marr_Hidreth_test():
	img_path = "./img/lena.jpg"
	"""
		feel free to modify these three parameters.
		 notice that the size should be an odd number.
	"""
	sigma = 1
	size = 5
	threshold_p = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

	for i in range(6):
		plt.subplot(2, 3, 1 + i)
		plt.title("threshold-%s" % threshold_p[i]), plt.xticks([]), plt.yticks([])
		edge = Marr_Hidreth_detect(img_path, threshold_p[i], sigma, size)
		plt.imshow(edge, cmap="gray")

	plt.show()


if __name__ == '__main__':
	Marr_Hidreth_test()
