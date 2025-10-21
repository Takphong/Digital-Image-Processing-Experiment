#Takphong 6601023610047

import cv2 as project

img = project.imread('awyYzxr_460s.jpg')
project.imshow('My Image Window', img)

k = project.waitKey(0)

if k == ord('s'):
    project.imwrite('Takphong.jpg', img)
    print("Image saved as 'saved_image.jpg'")
    project.destroyAllWindows()
else:
    print("Window closed without saving.")
    project.destroyAllWindows()