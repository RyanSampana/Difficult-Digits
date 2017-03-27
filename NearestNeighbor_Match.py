import numpy as np
import cv2
from matplotlib import pylab as pt
from matplotlib import pyplot as plt

from scipy import ndimage

import copy

from multiprocessing import Pool
from contextlib import closing

from functools import partial 

#Checks the overlap between two boundary boxes
def overlap(rect1, rect2):
	XA1 = rect1[0]
	XB1 = rect2[0]
	XA2 = rect1[0]+28
	XB2 = rect2[0]+28

	YA1 = rect1[1]
	YB1 = rect2[1]
	YA2 = rect1[1]+28
	YB2 = rect2[1]+28

	SI = max(0, min(XA2, XB2) - max(XA1, XB1)) * max(0, min(YA2, YB2) - max(YA1, YB1))

	return SI/(1568.0-SI) #1568 = 28*28*2

def post_process(dim, image,template):
	x1 = max(0,dim[0]-3)
	x2 = min(60,dim[0]+31)
	y1 = max(0,dim[1])
	y2 = min(60,dim[1]+28)
	img = image[y1:y2,x1:x2]
	crop(img,template)
	clean(img)
	center(img)
	return img

def crop(image,template):
	stop_top = False
	stop_bot = False
	for x in xrange(14):
		for y in xrange(28):
			if template[x][y] != 0:
				stop_top = True
			if template[27-x][y] != 0:
				stop_bot = True
		if not stop_top:
			for y_image in xrange(len(image[0])):
				image[x][y_image] = 0
		if not stop_bot:
			for y_image in xrange(len(image[0])):
				image[27-x][y_image] = 0
		if stop_top and stop_bot:
			break

def clean(image):
	for x in xrange(len(image)):
		for y in xrange(len(image[0])):
			if image[x][y] == 255:
				size = component_size(image,x,y)
				if size >= 255: #Edge case
					size == 254
				if size == 100:
					size = 99 
				image[image == 100] = size

	max_component = np.max(image)
	image[image < max_component] = 0
	image[image >= max_component] = 255

def component_size(image,x,y):
	if x >= len(image) or y >= len(image[0]) or x < 0 or y < 0:
		return 0
	if image[x][y] == 255:
		image[x][y] = 100
		return component_size(image,x+1,y)+component_size(image,x,y+1)+component_size(image,x-1,y)+component_size(image,x,y-1)+1
	else:
		return 0

def center(image):
	img = copy.deepcopy(image)
	a = np.where(img == 255)
	bbox = img[np.min(a[0]):np.max(a[0]), np.min(a[1]):np.max(a[1])]

	image[image != 0] = 0

	h = (len(image)-len(bbox))/2
	w = (len(image[0])-len(bbox[0]))/2

	for x in xrange(len(bbox)):
		for y in xrange(len(bbox[0])):
			image[x+h][y+w] = bbox[x][y]

def find_digits(test_x, mnist_train, train_y, pic):	
	
	# Prints every 10 iterations to keep sanity in check
	if pic % 10 == 0:
		print pic 

	image = test_x[pic]
	
	# Thresholding based on mean and standard deviation
	mean = cv2.meanStdDev(image)[0]
	std = cv2.meanStdDev(image)[1]
	if (mean > 220): 
		ret,image = cv2.threshold(image,min(mean+std+5,250),255,cv2.THRESH_BINARY)
	else:
		ret,image = cv2.threshold(image,min(max(mean+2*std,150),245),255,cv2.THRESH_BINARY)

	# Initialization - Most of this will be overrideden on the first iteration
	min_dist1 = 10000000 
	min_label1 = 4
	min_rect1 = [100,100]
	best1 = []
	mnist1 = -1

	min_dist2 = 10000000 
	min_label2 = 4
	min_rect2 = [100,100]
	best2 = []
	mnist2 = -1

	for i in xrange(len(mnist_train)): 
		label = mnist_train[i][0]
		template = mnist_train[i][1:]
		template = template.astype(np.uint8).reshape(28,28)

		#Find best match
		result = cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)
		min_val, max_val, min_loc, pt = cv2.minMaxLoc(result)

		dist = -result[pt[1]][pt[0]] #Take the negative so we can a minimum

		if dist < -0.5 and dist < min_dist2:
			if dist < min_dist1:
				if overlap(pt,min_rect1) < 0.55:
					min_dist2 = min_dist1
					min_label2 = min_label1
					min_rect2 = min_rect1
					#best2 = best1
					mnist2 = mnist1

				min_dist1 = dist
				min_label1 = label
				min_rect1 = pt
				best1 = template
				mnist1 = i

			elif overlap(pt,min_rect1) < 0.55:
				min_dist2 = dist
				min_label2 = label
				min_rect2 = pt
				#best2 = template
				mnist2 = i

		if min_dist2 < -0.90:
			#return min_label1+min_label2
			break
	
	# If we are uncertain about one of the digits, 
	# we remove the first one from the image, and try again. 
	CONSTANT = 0
	if min_dist2 > -0.80:
		image = copy.deepcopy(test_x[pic])
		ret,image = cv2.threshold(image,220,255,cv2.THRESH_BINARY) 
		#Remove best1 from the image
		a = np.where(best1 != 0)
		for x in xrange(len(image)):
			for y in xrange(len(image)):
				if x > np.min(a[0])+min_rect1[1]+CONSTANT and x < np.max(a[0])+min_rect1[1]-CONSTANT and y > np.min(a[1])+min_rect1[0]+CONSTANT and y < np.max(a[1])+min_rect1[0]-CONSTANT:
					if best1[x-min_rect1[1]][y-min_rect1[0]] != 0:
						image[x][y] = 0
		#clean(image) #remove particles
		#Retest
		for i in xrange(len(mnist_train)): 
			label = mnist_train[i][0]
			template = mnist_train[i][1:]
			template = template.astype(np.uint8).reshape(28,28)

			result = cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)
			min_val, max_val, min_loc, pt = cv2.minMaxLoc(result)
				
			dist = -result[pt[1]][pt[0]] #Take the negative so we can a minimum
			
			if dist < min_dist2 and overlap(pt,min_rect1) < 0.55:
				min_dist2 = dist
				min_label2 = label
				min_rect2 = pt
				#best2 = template
				mnist2 = i
			
			if min_dist2 < -0.90:
				#return min_label1+min_label2
				break

	return [mnist1,mnist2]

if __name__ == '__main__':

	# train_x = np.fromfile('train_x.bin', dtype='uint8',count=10000*3600)
	# train_x = train_x.reshape((10000,60,60))
	test_x = np.fromfile('test_x.bin', dtype='uint8')
	test_x = test_x.reshape((20000,60,60))
	train_y = np.genfromtxt('train_y.csv', delimiter=',', skip_header=1)
	mnist_trn = np.genfromtxt('mnist_train.csv', delimiter=',',max_rows=60000)
	mnist_test = np.genfromtxt('mnist_test.csv', delimiter=',',max_rows=10000)
	mnist_train = np.concatenate([mnist_trn,mnist_test])

	print '##### Data Loaded #####'

	func = partial(find_digits, test_x, mnist_train, train_y)

########### SAVE TO FILE ################

	f = open('mnist_matches_5k.csv','w')
	with closing(Pool()) as p:
		matches = p.map(func,range(5000))
	for pred in matches:
		f.write("%d,%d\n" % (pred[0],pred[1])) 
	f.close() 

########### ACCURACY ##############
	# correct = 0.0
	# for i in xrange(len(predictions)):
	# 	if predictions[i] == train_y[i+3000][1]:
	# 		correct += 1
	# 	else:
	# 		print i

	# accuracy = correct/len(predictions)
	# print accuracy

	print '##### Program Completed #####'


########### Display ##############
	# bottom_right = (min_rect1[0] + 28, min_rect1[1] + 28)
	# cv2.rectangle(image,min_rect1, bottom_right, 100, 1)

	# bottom_right = (min_rect2[0] + 28, min_rect2[1] + 28)
	# cv2.rectangle(image,min_rect2, bottom_right, 200, 1)

	# plt.subplot(331),plt.imshow(image,cmap = 'gray')
	# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(332),plt.imshow(best1,cmap = 'gray')
	# plt.title('Best Match 1 (Darker)'), plt.xticks([]), plt.yticks([])
	# plt.subplot(333),plt.imshow(best2,cmap = 'gray')
	# plt.title('Best Match 2 (Lighter)'), plt.xticks([]), plt.yticks([])
	# plt.show()