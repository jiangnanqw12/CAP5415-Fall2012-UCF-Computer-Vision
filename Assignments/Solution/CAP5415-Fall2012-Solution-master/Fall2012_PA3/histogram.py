# -*- coding: utf-8 -*-
"""
    @Author  : LiuZhian
    @Time    : 2019/8/29 0029 下午 3:19
    @Comment : 
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def calc_each_channel_hist(img):
	"""
	get histograms of each channel of given image.

	Using OpenCV build-in function
	:param img:
	:return:
	"""
	rows, cols = img.shape[:2]
	# create a mask, mask out every 50 pixels along the horizontal and vertical border
	mask = np.zeros(img.shape[:2], np.uint8)
	mask[50:rows - 50, 50:cols - 50] = 255
	masked_img = cv2.bitwise_and(img, img, mask=mask)

	color = ('b', 'g', 'r')

	plt.subplot(2, 2, 1)
	plt.title("origin-img")
	plt.imshow(img[:, :, ::-1])

	plt.subplot(2, 2, 2)
	plt.title("origin-img histogram")
	plt.ylabel("percentage(%)")
	plt.xlim([0, 256])
	for i, c in enumerate(color):
		hist = cv2.calcHist([img], [i], None, [256], [0, 256])
		hist = hist / img.shape[0] / img.shape[1] * 100
		plt.plot(hist, color=c)

	plt.subplot(2, 2, 3)
	plt.title("masked-img")
	plt.imshow(masked_img[:, :, ::-1])

	plt.subplot(2, 2, 4)
	plt.title("masked-img histogram")
	plt.ylabel("percentage(%)")
	plt.xlim([0, 256])
	for i, c in enumerate(color):
		hist = cv2.calcHist([img], [i], mask, [256], [0, 256])
		# 计算百分数
		hist = hist / img.shape[0] / img.shape[1] * 100
		plt.plot(hist, color=c)

	plt.show()


def get_hist(img):
	"""
	Calculate normalized histogram of given image.
	The image should be a 8-bit depth image.
	:param img:
	:return: a list type histogram
	"""
	hist = [0] * 256
	rows, cols = img.shape[:2]
	for r in range(rows):
		for c in range(cols):
			hist[img[r, c]] += 1

	hist = np.array(hist)
	hist = hist / (rows * cols)

	return hist


def hist_equ(img):
	"""
	Apply histogram equalization algorithm to enhance the contrast.

	:param img:
	:return: refined image
	"""
	# get the histogram first.
	hist = get_hist(img)

	# calculate the cumulative probability
	cp = np.array([sum(hist[:i + 1]) for i in range(len(hist))])

	# get transferred value by multiply L-1  (255 here)
	# after that, we floor rounding to integer value.
	values = np.uint8(cp * 255)

	rows, cols = img.shape[:2]
	res = np.zeros((rows, cols), np.uint8)
	for r in range(rows):
		for c in range(cols):
			res[r, c] = values[img[r, c]]

	return res


def test_my_hist_equ(img):
	origin_hist = get_hist(img)
	enhanced_img = hist_equ(img)
	enhanced_hist = get_hist(enhanced_img)
	cv2.imwrite("hist-equ.jpg", enhanced_img)

	openCV_res = cv2.equalizeHist(img)
	openCV_res_hist = get_hist(openCV_res)
	cv2.imwrite('opencv-hist-equ.jpg', openCV_res)

	plt.figure(figsize=(8, 8))

	plt.subplot(3, 2, 1)
	plt.title("input")
	plt.imshow(img, cmap='gray')

	plt.subplot(3, 2, 2)
	plt.title("input histogram")
	plt.plot(origin_hist)

	plt.subplot(3, 2, 3)
	plt.title("enhanced image")
	plt.imshow(enhanced_img, cmap='gray')

	plt.subplot(3, 2, 4)
	plt.title("enhanced image histogram")
	plt.plot(enhanced_hist)

	plt.subplot(3, 2, 5)
	plt.title("OpenCV enhanced image")
	plt.imshow(openCV_res, cmap='gray')

	plt.subplot(3, 2, 6)
	plt.title("OpenCV enhanced image histogram")
	plt.plot(openCV_res_hist)

	plt.show()


if __name__ == '__main__':
	img = cv2.imread("./x-ray.jpg", cv2.IMREAD_GRAYSCALE)
