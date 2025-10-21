 import cv2
 import numpy as np
 
 
 def custom_median_filter(gray, ksize=5):
    pad = ksize // 2
    padded = np.pad(gray, pad, mode='edge')
    output = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            region = padded[i:i+ksize, j:j+ksize].flatten()
            output[i, j] = np.median(region)
    return output


 def unsharp_mask(gray, blur_ksize=5, amount=1.5):
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    mask = cv2.subtract(gray, blurred)
    sharpened = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
    return blurred, mask, sharpened
 cap = cv2.VideoCapture(0)
 
 
 if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()
 print("Press SPACE to capture")
 while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    frame = cv2.flip(frame, 1)
    cv2.imshow("Webcam (Press SPACE to capture)", frame)
    key = cv2.waitKey(1)
    if key == ord(' '):
        captured = frame.copy()
        break
    
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
        
        
 cap.release()
 cv2.destroyAllWindows()
 gray = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)
 median_result = custom_median_filter(gray, 5)
 n = 7
 avg_kernel = np.ones((n, n), np.float32) / (n * n)
 avg_result = cv2.filter2D(captured, -1, avg_kernel)
 lap_kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]])
 laplacian = cv2.filter2D(gray, cv2.CV_64F, lap_kernel)
 laplacian = cv2.convertScaleAbs(laplacian)
 blurred, detail_mask, sharpened = unsharp_mask(gray)
cv2.imshow("Original image", captured)
 cv2.imshow("Grayscale image", gray)
 cv2.imshow("1. Median Filter by (5x5)", median_result)
 cv2.imshow("2. Averaging Filter by (7x7)", avg_result)
 cv2.imshow("3. Laplacian", laplacian)
 cv2.imshow("4a. Blurred (Unsharp)", blurred)
 cv2.imshow("4b. Detail Mask", detail_mask)
 cv2.imshow("4c. Final Sharpened", sharpened)
 print("Done. Press any key to close.")
 cv2.waitKey(0)
 cv2.destroyAllWindows()