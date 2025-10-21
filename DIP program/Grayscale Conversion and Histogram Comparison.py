import cv2 as project
import matplotlib.pyplot as mp
import numpy as np

img = project.imread('C:/Users/Admin/.spyder-py3/Crop1.png')
project.imshow('Original Image', img)

R = img[:, :, 2]
G = img[:, :, 1]
B = img[:, :, 0]

SA = (R + G + B) / 3
test1 = SA.astype(np.uint8)

WA = 0.299 * R + 0.587 * G + 0.114 * B
test2 = WA.astype(np.uint8)

project.imshow('Simple Average', test1)
project.imshow('Weighted Average', test2)
project.waitKey(0)
project.destroyAllWindows()

mp.figure('Histogram')
mp.subplot(121)
mp.hist(test1.ravel(), 256, [0, 256])
mp.title('Simple Average')
mp.subplot(122)
mp.hist(test2.ravel(), 256, [0, 256])
mp.title('Weighted Average')
mp.show()