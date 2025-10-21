import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('C:/Users/Admin/.spyder-py3/Crop1.png')
rows, cols, ch = img.shape  # get the size of image

# need 4 point for original input (x,y) for top-left, top-right, bottom-left, bottom-right
pts1 = np.float32([ [130, 180],
                    [729, 179],
                    [80, 490],
                    [900, 445] ])

# need 4 point for new position(x,y) for top-left, top-right, bottom-left, bottom-right
pts2 = np.float32([ [0, 0],
                    [300, 0],
                    [0, 300],
                    [300, 300] ])

M = cv.getPerspectiveTransform(pts1, pts2)  # calculate the transformation matrix
dst = cv.warpPerspective(img, M, (300, 300))

plt.subplot(121)
plt.title('input')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

plt.subplot(122)
plt.title('output')
plt.imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()