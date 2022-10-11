# -*- coding: utf-8 -*-
"""
    @Author  : LiuZhian
    @Time    : 2019/8/24 0024 下午 3:42
    @Comment : 
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy.linalg as LA

from utils import gaussian_filter


def generate_gaussian_octave(input, s, sigma):
	"""
		The initial image is incrementally convolved with Gaussian
		to produce images separated by a constant factor k in scale space.
		i.e. k=2**(1/s), where s is the number of images we want in each DoG octave.

	:param input: input image in a specific octave
	:param s: number of images in each DoG octave (5 was chosen in the paper)
	:param sigma: prior smoothing for each octave (1.6 was chosen in the paper)
	:return:
	"""
	octave = [input]
	k = 2 ** (1 / s)
	g_kernel = gaussian_filter(k * sigma)

	for i in range(s + 2):
		next_layer = cv2.filter2D(octave[-1], -1, g_kernel)
		octave.append(next_layer)

	return octave


def generate_DoG_octave(gaussian_octave):
	"""
	generate DoG octave using some gaussian blurred images under different scales
	:param gaussian_octave: gaussian octave
	:return:
	"""
	octave = []
	for i in range(len(gaussian_octave) - 1):
		octave.append(gaussian_octave[i + 1] - gaussian_octave[i])

	# add an new axis for simply finding extrema
	# octave shape (layers_num_per_octave,rows,cols)
	octave = np.concatenate([o[np.newaxis, :, :] for o in octave], axis=0)

	return octave


def generate_gaussian_pyramids(img, octave_num, s=5, sigma=1.6):
	"""
	generate gaussian pyramids using some  successive gaussian octaves.
	:param img: source image
	:param octave_num:
	:param s: the number of images we want in each DoG octave
	:param sigma:
	:return:
	"""
	pyramids = []
	for i in range(octave_num):
		cur_octave = generate_gaussian_octave(img, s, sigma)
		pyramids.append(cur_octave)
		# here, we use the third to last image as the initial input
		# for the next level of octave
		img = cv2.pyrDown(cur_octave[-3])

	return pyramids


def generate_DoG_pyramids(gaussian_pyramids):
	pyramids = []
	for gaussian_octave in gaussian_pyramids:
		pyramids.append(generate_DoG_octave(gaussian_octave))
	return pyramids


def get_candidate_keypoints(DoG_octaves):
	candidate = []
	for layer in range(1, DoG_octaves.shape[0] - 1):
		for i in range(1, DoG_octaves.shape[1] - 1):
			for j in range(1, DoG_octaves.shape[1] - 1):
				patch = DoG_octaves[layer - 1:layer + 2, i - 1:i + 2, j - 1:j + 2]
				# if the central pixel is the extrema
				if np.argmax(patch) == 13 or np.argmin(patch) == 13:
					candidate.append([layer, i, j])

	return candidate


def get_partial_derivatives(D, s, x, y):
	"""
	calculate the first and second partial derivatives to be used in keypoints localization.
	:param D: DoG octave with shape (layers_num_per_octave,rows,cols)
	:param s: scale
	:param x: x-coordinate i.e. col index
	:param y: y-coordinate i.e. row index
	:return:
	"""
	dx = D[s, y, x + 1] - D[s, y, x]
	dy = D[s, y + 1, x] - D[s, y, x]
	ds = D[s + 1, y, x] - D[s, y, x]

	dxx = D[s, y, x + 1] - 2 * D[s, y, x] + D[s, x - 1, y]  # d(s,x,y)-d(s,x-1,y)
	dyy = D[s, y + 1, x] - 2 * D[s, y, x] + D[s, y, x - 1]  # d(s,x,y)-d(s,x,y-1)
	dss = D[s + 1, y, x] - 2 * D[s, y, x] + D[s - 1, y, x]  # d(s,x,y)-d(s-1,x,y)
	dxy = (D[s, y, x + 1] - D[s, y, x]) - (D[s, y - 1, x + 1] - D[s, y - 1, x])  # d(s,x,y)-d(s,x,y-1)
	dxs = (D[s, y, x + 1] - D[s, y, x]) - (D[s - 1, y, x + 1] - D[s - 1, y, x])  # d(s,x,y)-d(s-1,x,y)
	dys = (D[s, y + 1, x] - D[s, y, x]) - (D[s - 1, y + 1, x] - D[s - 1, y, x])  # d(s,x,y)-d(s-1,x,y)

	# dx = (D[s, x + 1, y] - D[s, x - 1, y]) / 2
	# dy = (D[s, x, y + 1] - D[s, x, y - 1]) / 2
	# ds = (D[s + 1, x, y] - D[s - 1, x, y]) / 2
	#
	# dxx = D[s, y, x + 1] - 2 * D[s, y, x] + D[s, y, x - 1]
	# dyy = D[s, y + 1, x] - 2 * D[s, y, x] + D[s, y - 1, x]
	# dss = D[s + 1, y, x] - 2 * D[s, y, x, s] + D[s - 1, y, x]
	#
	# dxy = ((D[s, y + 1, x + 1] - D[s, y + 1, x - 1]) - (D[s, y - 1, x + 1] - D[s, y - 1, x - 1])) / 4
	# dxs = ((D[s + 1, y, x + 1] - D[s + 1, y, x - 1]) - (D[s - 1, y, x + 1] - D[s - 1, y, x - 1])) / 4.
	# dys = ((D[s + 1, y + 1, x] - D[s + 1, y - 1, x]) - (D[s - 1, y + 1, x] - D[s - 1, y - 1, x])) / 4.
	J = np.array([dx, dy, ds])
	HD = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
	offset = -LA.inv(HD).dot(J)
	return offset, J, HD[:2, :2], x, y, s


def sift_test():
	img = cv2.imread("./img/lena.jpg", cv2.IMREAD_GRAYSCALE)
	gaussian_pyr = generate_gaussian_pyramids(img, 3)
	DoG_pyr = generate_DoG_pyramids(gaussian_pyr)
	print(len(gaussian_pyr[0]))
	print(len(DoG_pyr[0]))

	for i in range(8):
		plt.subplot(3, 3, i + 1), plt.xticks([]), plt.yticks([])
		# plt.title("layer%d,shape:(%d,%d)" % (i + 1, gaussian_pyr[0][i].shape[0], gaussian_pyr[0][i].shape[1]))
		plt.title("layer%d" % (i + 1,))
		plt.imshow(gaussian_pyr[0][i], cmap="gray")
	plt.show()

	for i in range(7):
		plt.subplot(3, 3, i + 1), plt.xticks([]), plt.yticks([])
		# plt.title("layer%d,shape:(%d,%d)" % (i + 1, gaussian_pyr[0][i].shape[0], gaussian_pyr[0][i].shape[1]))
		plt.title("layer%d" % (i + 1,))
		plt.imshow(DoG_pyr[0][i, :, :], cmap="gray")
	plt.show()


def sift_openCV_test():
	gray = cv2.imread("./img/lena.jpg", cv2.IMREAD_GRAYSCALE)
	sift = cv2.xfeatures2d.SIFT_create()
	keypoints = sift.detect(gray)

	kp_img = cv2.drawKeypoints(gray, keypoints, gray, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imwrite('./result/sift_lena.jpg', kp_img)

	# kp, des = sift.compute(gray, keypoints)



if __name__ == '__main__':
	sift_openCV_test()
