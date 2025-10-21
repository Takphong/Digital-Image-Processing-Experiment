import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

img1 = cv2.imread('grass1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('grass2.jpg', cv2.IMREAD_GRAYSCALE)

img1 = (img1 / 32).astype(np.uint8)
img2 = (img2 / 32).astype(np.uint8)

distances = [1]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

glcm1 = graycomatrix(img1, distances, angles, levels=8, symmetric=True, normed=True)
glcm2 = graycomatrix(img2, distances, angles, levels=8, symmetric=True, normed=True)

props = ['contrast', 'correlation', 'energy', 'homogeneity']

print("Grass 1 Properties:")
for p in props:
    print(f"{p}: {graycoprops(glcm1, p).mean():.4f}")

print("\nGrass 2 Properties:")
for p in props:
    print(f"{p}: {graycoprops(glcm2, p).mean():.4f}")