import cv2

def nothing(x):
    pass

real = cv2.VideoCapture(0)
cv2.namedWindow("canny")
cv2.createTrackbar("th1", "canny", 0, 255, nothing)
cv2.createTrackbar("th2", "canny", 0, 255, nothing)

while True:
    ret, frame = real.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th1 = cv2.getTrackbarPos("th1", "canny")
    th2 = cv2.getTrackbarPos("th2", "canny")
    edges = cv2.Canny(gray, th1, th2)
    frame = cv2.flip(frame, 1)
    edges = cv2.flip(edges, 1)
    cv2.imshow("canny", edges)
    cv2.imshow("Original", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

real.release()
cv2.destroyAllWindows()