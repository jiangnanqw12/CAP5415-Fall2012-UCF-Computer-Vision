# -*- coding: utf-8 -*-
"""
    @Author  : LiuZhian
    @Time    : 2019/8/21 0021 下午 8:07
    @Comment : 
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse


def gaussian_and_laplacian_pyramid_test():
	g1 = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
	g2 = cv2.pyrDown(g1)
	g3 = cv2.pyrDown(g2)
	g4 = cv2.pyrDown(g3)
	g5 = cv2.pyrDown(g4)
	g6 = cv2.pyrDown(g5)
	gaussian_pyrimads = [g1, g2, g3, g4, g5, g6]
	for g in gaussian_pyrimads:
		print(g.shape)

	l1 = g1 - cv2.pyrUp(g2)
	l2 = g2 - cv2.pyrUp(g3, dstsize=g2.shape[::-1])
	l3 = g3 - cv2.pyrUp(g4, dstsize=g3.shape[::-1])
	l4 = g4 - cv2.pyrUp(g5, dstsize=g4.shape[::-1])
	l5 = g5 - cv2.pyrUp(g6, dstsize=g5.shape[::-1])
	laplacian_pyrimads = [l1, l2, l3, l4, l5, g6]

	for i in range(6):
		plt.subplot(2, 3, i + 1)
		plt.imshow(gaussian_pyrimads[i][:, :], cmap="gray")
	plt.show()

	for i in range(6):
		plt.subplot(2, 3, i + 1)
		plt.imshow(laplacian_pyrimads[i][:, :], cmap="gray")
	plt.show()


def img_blending_using_pyramid(imgA, imgB, layer=6):
	"""
	Done as follows:
		1. Load the two images of apple and orange
		2. Find the Gaussian Pyramids for apple and orange (in this particular example, number of levels is 6)
		3. From Gaussian Pyramids, find their Laplacian Pyramids
		4. Now join the left half of apple and right half of orange in each levels of Laplacian Pyramids
		5. Finally from this joint image pyramids, reconstruct the original image.
	:param imgA: path of img A
	:param imgB: path of img B
	:param layer:  total layers of the pyramids
	:return:
	"""

	# step 1
	A = cv2.imread(imgA)
	B = cv2.imread(imgB)

	# step 2
	# 生成A图的高斯金字塔
	tmp = A.copy()
	gpa = [tmp]
	for i in range(layer):
		tmp = cv2.pyrDown(tmp)
		gpa.append(tmp)
	# 生成B图的高斯金字塔
	tmp = B.copy()
	gpb = [tmp]
	for i in range(layer):
		tmp = cv2.pyrDown(tmp)
		gpb.append(tmp)

	# step 3
	# 生成A图的拉普拉斯金字塔  (注意，高层的拉普拉斯金字塔在数组前面)
	lpA = [gpa[layer - 1]]  # 最后一层拉普拉斯设置为对应的高斯层
	for i in range(layer - 1, 0, -1):  # 45,52  90,103
		L = gpa[i - 1] - cv2.pyrUp(gpa[i], dstsize=(gpa[i - 1].shape[1], gpa[i - 1].shape[0]))
		lpA.append(L)
	# 生成B图的拉普拉斯金字塔
	lpB = [gpb[layer - 1]]  # 最后一层拉普拉斯设置为对应的高斯层
	for i in range(layer - 1, 0, -1):
		L = gpb[i - 1] - cv2.pyrUp(gpb[i], dstsize=(gpb[i - 1].shape[1], gpb[i - 1].shape[0]))
		lpB.append(L)

	# step 4
	l_mixed = []
	for la, lb in zip(lpA, lpB):
		rows, cols, channel = la.shape
		l_item = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
		# l_item = np.vstack((la[0:rows // 2, :], lb[:rows // 2, :]))
		# l_item = (la + lb) / 2
		l_mixed.append(l_item)

	# step 5 重建图片
	generated_res = [l_mixed[0]]
	generate_each = l_mixed[0]
	for i in range(1, layer):
		generate_each = cv2.pyrUp(generate_each, dstsize=(l_mixed[i].shape[1],l_mixed[i].shape[0])) + l_mixed[i]
		generated_res.append(generate_each)

	# 输出并保存到图片
	# 直接相连
	height, width, c = A.shape
	direct_blending = np.hstack((A[:, 0: width // 2], B[:, width // 2:]))

	cv2.imwrite('result/Pyramid_blending-layer%s.jpg'%layer, generate_each)
	cv2.imwrite('result/Direct_blending.jpg', direct_blending)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--imgA", type=str, default="img/apple.jpg", help="full path of image A")
	parser.add_argument("--imgB", type=str, default="img/orange.jpg", help="full path of image B")
	parser.add_argument("--layer", type=int, default=6, help="total layers of the pyramids")

	imgA = parser.parse_args().imgA
	imgB = parser.parse_args().imgB
	layer = parser.parse_args().layer

	img_blending_using_pyramid(imgA, imgB, layer)
