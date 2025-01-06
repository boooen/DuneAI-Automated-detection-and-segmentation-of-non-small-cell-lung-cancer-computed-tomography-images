# -*- coding: utf-8 -*-
"""
Spyder Editor
s.primakov@maastrichtuniversity.nl
This is a temporary script file.
"""

import numpy as np
import SimpleITK as sitk
import cv2
import os
import matplotlib.pyplot as plt
from skimage.measure import label,regionprops
import matplotlib.patches as patches
#Keras & TF
import keras
import keras.backend as K
from keras.models import load_model
import math, random

# 数据约定：一个点是一对浮点数 (x, y)。一个圆是一个三元组浮点数 (中心 x, 中心 y, 半径)。

# 返回包含所有给定点的最小圆。以预期的 O(n) 时间运行，已随机化。
# 输入：一系列浮点数或整数对，例如 [(0,5), (3.1,-2.7)]。
# 输出：表示圆的三元组浮点数。
# 注意：如果给出 0 个点，则返回 None。如果给出 1 个点，则返回半径为 0 的圆。
#
# 初始状态：没有已知的边界点
def make_circle(points):
	# 转换为浮点数并打乱顺序
	shuffled = [(float(x), float(y)) for (x, y) in points]
	random.shuffle(shuffled)
	
	# 逐步将点添加到圆或重新计算圆
	c = None
	for (i, p) in enumerate(shuffled):
		if c is None or not is_in_circle(c, p):
			c = _make_circle_one_point(shuffled[ : i + 1], p)
	return c

# 已知一个边界点
def _make_circle_one_point(points, p):
	c = (p[0], p[1], 0.0)
	for (i, q) in enumerate(points):
		if not is_in_circle(c, q):
			if c[2] == 0.0:
				c = make_diameter(p, q)
			else:
				c = _make_circle_two_points(points[ : i + 1], p, q)
	return c

# 已知两个边界点
def _make_circle_two_points(points, p, q):
	circ = make_diameter(p, q)
	left  = None
	right = None
	px, py = p
	qx, qy = q
	
	# 对于不在两点圆中的每个点
	for r in points:
		if is_in_circle(circ, r):
			continue
		
		# 形成一个外接圆并将其分类在左侧或右侧
		cross = _cross_product(px, py, qx, qy, r[0], r[1])
		c = make_circumcircle(p, q, r)
		if c is None:
			continue
		elif cross > 0.0 and (left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0], left[1])):
			left = c
		elif cross < 0.0 and (right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0], right[1])):
			right = c
	
	# 选择要返回的圆
	if left is None and right is None:
		return circ
	elif left is None:
		return right
	elif right is None:
		return left
	else:
		return left if (left[2] <= right[2]) else right

def make_diameter(a, b):
	cx = (a[0] + b[0]) / 2.0
	cy = (a[1] + b[1]) / 2.0
	r0 = math.hypot(cx - a[0], cy - a[1])
	r1 = math.hypot(cx - b[0], cy - b[1])
	return (cx, cy, max(r0, r1))

def make_circumcircle(a, b, c):
	# 来自维基百科的数学算法：外接圆
	ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2.0
	oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2.0
	ax = a[0] - ox;  ay = a[1] - oy
	bx = b[0] - ox;  by = b[1] - oy
	cx = c[0] - ox;  cy = c[1] - oy
	d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
	if d == 0.0:
		return None
	x = ox + ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
	y = oy + ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
	ra = math.hypot(x - a[0], y - a[1])
	rb = math.hypot(x - b[0], y - b[1])
	rc = math.hypot(x - c[0], y - c[1])
	return (x, y, max(ra, rb, rc))

_MULTIPLICATIVE_EPSILON = 1 + 1e-14

def is_in_circle(c, p):
	return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON

# 返回由 (x0, y0)、(x1, y1)、(x2, y2) 定义的三角形的有符号面积的两倍。
def _cross_product(x0, y0, x1, y1, x2, y2):
	return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

def calculate_values(mask_array_predicted,params,mask_array_orig=None):
    # 计算 RECIST 值（肿瘤最长直径）和肿瘤体积
    # 核心逻辑：遍历所有切片，找出肿瘤直径最大的切片，并计算其最小外接圆。
    
    recist,idx = 0,0
    # 遍历预测的掩膜数组中的每个切片
    for i,temp_slice in enumerate(mask_array_predicted):
        if np.sum(temp_slice.flatten())>10:
            x_ind,y_ind = np.where(temp_slice==1)
            # 使用最小外接圆算法计算包含所有肿瘤像素的最小圆
            circ = make_circle(zip(x_ind,y_ind))
            # 计算该切片的肿瘤直径（像素数量）
            slice_temp_diameter = int(2*circ[2]*params[0])

            if slice_temp_diameter>recist:
                recist = slice_temp_diameter
                idx =i
                circle = circ

    volume_predicted = np.round(np.sum(mask_array_predicted.flatten())*params[0]*params[1]*params[2]*0.001,2)
    
    # 如果提供了原始掩膜数组，则计算原始肿瘤的 RECIST 值和体积
    if mask_array_orig:
        recist_orig,idx = 0,0
        mask_array_orig = np.squeeze(mask_array_orig)
        for i,temp_slice in enumerate(mask_array_orig):
            if np.sum(temp_slice.flatten())>10:
                x_ind,y_ind = np.where(temp_slice==1)
                circ = make_circle(zip(x_ind,y_ind))
                slice_temp_diameter = int(2*circ[2]*params[0])
                if slice_temp_diameter>recist_orig:
                    recist_orig = slice_temp_diameter
                    idx=i
                    circle = circ
                    
        volume_orig = np.round(np.sum(mask_array_orig.flatten())*params[0]*params[1]*params[2]*0.001,2)
        return recist,recist_orig,volume_predicted,volume_orig,idx
    else:
        return recist,volume_predicted,idx,circ
        
