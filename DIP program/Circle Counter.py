import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img_path = "many_cir.jpg"
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

# Ensure binary
_, bin_img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# Morphology to remove noise
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
clean = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel)

# Separate touching circles
sep_k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (35,35))
separated = cv.erode(clean, sep_k, iterations=1)

# Connected component labeling
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(separated, connectivity=8)

# Filter tiny blobs (noise)
H, W = separated.shape
min_area = int(0.01 * H * W)
valid_ids = [i for i in range(1, num_labels) if stats[i, cv.CC_STAT_AREA] >= min_area]
count = len(valid_ids)

# Annotate
annot = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
for i in valid_ids:
    x, y, w, h, area = stats[i]
    cx, cy = centroids[i]
    cv.rectangle(annot, (x,y), (x+w,y+h), (0,255,0), 2)
    cv.circle(annot, (int(cx), int(cy)), 5, (0,0,255), -1)
    cv.putText(annot, f"ID:{i}", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

# Visualization
plt.figure(figsize=(12,6))
plt.subplot(1,3,1); plt.imshow(bin_img, cmap='gray'); plt.title("Binary"); plt.axis("off")
plt.subplot(1,3,2); plt.imshow(separated, cmap='gray'); plt.title("Separated"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(cv.cvtColor(annot, cv.COLOR_BGR2RGB)); plt.title(f"Count = {count}"); plt.axis("off")
plt.tight_layout()
plt.show()
