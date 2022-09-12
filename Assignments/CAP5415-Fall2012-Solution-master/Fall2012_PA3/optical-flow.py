# -*- coding: utf-8 -*-
"""
    @Author  : LiuZhian
    @Time    : 2019/8/27 0027 下午 5:25
    @Comment : 
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def Lucas_Kanade(img1, img2, window_size, tau=0.01):
	"""
	Lucas Kanade Algorithm
	:param img1: 	input gray image one
	:param img2:	input gray image two
	:param window_size: sliding window size (e.g. 3 by 3)
	:return: velocity U,V as dx/dt and dy/dt respectively.
	"""
	assert (img1.shape == img2.shape)
	assert (window_size % 2 == 1)  # assume window_size is an odd number

	# Normalize the images
	img1 = img1 / 255.
	img2 = img2 / 255.

	U = np.zeros(img1.shape)
	V = np.zeros(img1.shape)

	kernel_x = np.array([[-1, 1], [-1, 1]])
	kernel_y = np.array([[-1, -1], [1, 1]])
	kernel_t = np.array([[-1, -1], [-1, -1]])

	fx = cv2.filter2D(img1, cv2.CV_64F, kernel_x)
	fy = cv2.filter2D(img1, cv2.CV_64F, kernel_y)
	ft = cv2.filter2D(img1, cv2.CV_64F, kernel_t) + \
		 cv2.filter2D(img2, cv2.CV_64F, -1 * kernel_t)

	# plt.subplot(2, 2, 1), plt.imshow(fx, cmap="gray")
	# plt.subplot(2, 2, 2), plt.imshow(fy, cmap="gray")
	# plt.subplot(2, 2, 3), plt.imshow(ft, cmap="gray")
	# plt.show()

	# refer to Devin's implementation
	# https://sandipanweb.wordpress.com/2018/02/25/implementing-lucas-kanade-optical-flow-algorithm-in-python/
	w = window_size // 2
	rows, cols = img1.shape
	for i in range(w, rows - w):
		for j in range(w, cols - w):
			Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
			Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
			It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()

			# get b here
			b = np.reshape(It, (It.shape[0], 1))
			# get A here
			A = np.vstack((Ix, Iy)).T

			# if threshold τ is larger than the smallest eigenvalue of A'A:
			tmp = np.min(abs(np.linalg.eigvals(np.matmul(A.T, A))))
			if tmp >= tau:
				nu = np.matmul(np.linalg.pinv(A), b)  # get velocity here
				U[i, j] = nu[0]
				V[i, j] = nu[1]

	return U, V


def plot_optical_flow(img, U, V, t=10):
	"""
	Plots optical flow given U,V and one of the images
	:param firgure:
	:param img:
	:param U:
	:param V:
	:param t: 	plot one arrow form each t arrows.
				Change t if required, affects the number of arrows
	:return:
	"""

	# t should be between 1 and min(U.shape[0],U.shape[1])
	assert (t >= 1 and t <= min(U.shape[0], U.shape[1]))

	# Subsample U and V to get visually pleasing output
	U1 = U[::t, ::t]
	V1 = V[::t, ::t]

	# Create meshgrid of subsampled coordinates
	r, c = img.shape[0], img.shape[1]
	cols, rows = np.meshgrid(np.linspace(0, c - 1, c), np.linspace(0, r - 1, r))
	cols = cols[::t, ::t]
	rows = rows[::t, ::t]
	# Plot optical flow
	plt.figure(figsize=(10, 10))
	plt.imshow(img)
	plt.quiver(cols, rows, U1, V1, color="blue")


def visualize_optical_flow(img, optical_flow):
	"""
	 ref: https://stackoverflow.com/questions/10161351/opencv-how-to-plot-velocity-vectors-as-arrows-in-using-single-static-image
	!!! DO NOT USE THIS !!!
	:param img:
	:param optical_flow:
	:return:
	"""
	t = 10  # for reduce arrow numbers
	#  l = sqrt(velocity_x**2+velocity_y**2)
	l = np.sqrt(optical_flow[::t, ::t, 0] ** 2 + optical_flow[::t, ::t, 1] ** 2)
	# find max length
	max_l = np.max(l)

	rows, cols = img.shape[0:2]
	for i in range(0, rows, t):
		for j in range(0, cols, t):
			spinSize = l[i // t, j // t] / max_l * 5.0

			pt1 = (j, i)
			pt2 = (int(j + optical_flow[i, j, 0]), int(j + optical_flow[i, j, 1]))
			cv2.line(img, pt1, pt2, (0, 255, 0), 1)

			angle = np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
			pt1 = (int(pt2[0] + spinSize * np.cos(angle + np.pi / 4)),
				   int(pt2[1] + spinSize * np.sin(angle + np.pi / 4)))
			cv2.line(img, pt1, pt2, (0, 255, 0), 1)

			pt1 = (int(pt2[0] + spinSize * np.cos(angle - np.pi / 4)),
				   int(pt2[1] + spinSize * np.sin(angle - np.pi / 4)))
			cv2.line(img, pt1, pt2, (0, 255, 0), 1)

	return img


def keypoint_optical_flow(video_file=0):
	"""
	KeyPoints detector + Lucas-Kanade  Optical Flow
	:param video_file:  if this param is empty, use camera device as default
	:return:
	"""
	cap = cv2.VideoCapture(video_file)

	# params for ShiTomasi corner detection
	feature_params = dict(maxCorners=100,
						  qualityLevel=0.3,
						  minDistance=7,
						  blockSize=7)

	# Parameters for lucas kanade optical flow
	lk_params = dict(winSize=(15, 15),
					 maxLevel=2,
					 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	# color to draw flow track
	color = (0, 0, 255)

	# Take first frame and find corners in it
	ret, pre_frame = cap.read()
	prev_gray = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
	prevPts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

	# create a mask image for drawing purpose
	mask = np.zeros_like(pre_frame)

	while cap.isOpened():
		ret, cur_frame = cap.read()
		cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

		# calculate optical flow
		nextPts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prevPts, None, **lk_params)

		# Select good points
		good_new = nextPts[status == 1]
		good_old = prevPts[status == 1]

		# draw the tracks
		for i, (new, old) in enumerate(zip(good_new, good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			mask = cv2.line(mask, (a, b), (c, d), color, 2)
			cv2.circle(cur_frame, (a, b), 5, color, -1)

		img = cv2.add(cur_frame, mask)

		cv2.imshow('frame', img)

		k = cv2.waitKey(30) & 0xff
		if k == 27:  # esc key
			break

		# Updates previous frame
		prev_gray = cur_gray
		prevPts = nextPts.reshape(-1, 1, 2)

	cap.release()
	cv2.destroyAllWindows()


def dense_optical_flow(video_file=0):
	cv2.namedWindow("source", cv2.WINDOW_NORMAL)
	cv2.namedWindow("optical-hsv", cv2.WINDOW_NORMAL)

	cv2.resizeWindow('source', 800, 450)
	cv2.resizeWindow('optical-hsv', 800, 450)

	cap = cv2.VideoCapture(video_file)

	_, pre_frame = cap.read()
	pre_gray = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)

	hsv = np.zeros_like(pre_frame)
	hsv[:, :, 1] = 255

	while True:
		_, cur_frame = cap.read()
		cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

		optical_flow = cv2.calcOpticalFlowFarneback(pre_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)

		mag, ang = cv2.cartToPolar(optical_flow[:, :, 0], optical_flow[:, :, 1])
		hsv[:, :, 0] = ang * 180 / np.pi / 2
		hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
		rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

		cv2.imshow('source', pre_frame, )
		cv2.imshow('optical-hsv', rgb)
		k = cv2.waitKey(1) & 0xff
		if k == 27:
			break
		elif k == ord('s'):
			cv2.imwrite('./result/frame.png', cur_frame)
			cv2.imwrite('./result/optical-hsv.png', rgb)

		pre_gray = cur_gray
		pre_frame = cur_frame

	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	dense_optical_flow("./video/car_moving.mp4")
