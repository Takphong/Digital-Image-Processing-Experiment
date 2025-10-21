import cv2
import numpy as nu


img_gray = cv2.imread('awyYzxr_460s.jpg')
img_gray_f32 = img_gray.astype(nu.float32)
cv2.imshow('Image Enhancement', img_gray)


current_mode = 0
gamma = 1.0

#------------------------------------------------------------------------------

def change_img_enhancement(mode):
    global current_mode
    current_mode = mode
    
    
    if mode == 0: # Original
        cv2.imshow('Image Enhancement', img_gray)
    
    elif mode == 1: # Image Negatives
        L = 256
        r = img_gray
        img_negative = (L - 1) - r
        cv2.imshow('Image Enhancement', img_negative)
    
    elif mode == 2: # Log Transformations
        c = 255 / (nu.log(1 + nu.max(img_gray_f32)))
        r = img_gray_f32

        img_log = c * (nu.log(1 + r))
        img_log = img_log.astype(nu.uint8)
        cv2.imshow('Image Enhancement', img_log)

    elif mode == 3:
        c = 255
        r = img_gray / 255
        gamma = 0.2
    
        img_powerlaw = c * (r ** gamma)
        img_powerlaw = img_powerlaw.astype(nu.uint8)
        cv2.imshow('Image Enhancement', img_powerlaw)

#------------------------------------------------------------------------------

cv2.createTrackbar('Value', 'Image Enhancement', 0, 3, change_img_enhancement)
cv2.waitKey()
cv2.destroyAllWindows()