# -*- coding: utf-8 -*-
"""
    @Author  : LiuZhian
    @Time    : 2019/8/24 0024 下午 3:50
    @Comment : 
"""
import numpy as np


def gaussian_filter(sigma):
	size = 2 * np.ceil(3 * sigma) + 1
	x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
	gaussian = np.exp((-x ** 2 - y ** 2) / 2 / sigma ** 2) / 2 / sigma ** 2 / np.pi
	return gaussian
