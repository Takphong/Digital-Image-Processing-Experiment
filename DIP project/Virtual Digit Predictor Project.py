import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === STEP 1: LOAD THE PREDICTION MODEL ===
try:
    model = load_model("digit_identify.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure 'digit_identify.keras' is in the same folder.")
    exit()

# === STEP 2: REVISED PREDICTION FUNCTION ===
def predict_drawn_digit(image_array):

    gray_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray_img, (28, 28))

    img_norm = img_resized / 255.0

    img_input = img_norm.reshape(1, 28, 28, 1)

    pred = model.predict(img_input)
    digit = np.argmax(pred)
    confidence = np.max(pred) * 100
    
    return digit, confidence

# === STEP 3: SETUP FOR VIRTUAL DRAWING ===
cap = cv2.VideoCapture(0)

lower_color = np.array([100, 150, 50])  # 🦊 blue object
upper_color = np.array([140, 255, 255])

canvas = None
prediction_text = ""

# === STEP 4: MAIN REAL-TIME LOOP ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Flip frame

    if canvas is None:
        canvas = np.zeros_like(frame)

    # --- Color Tracking and Drawing ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:  # Filter out small noise
            (x, y, w, h) = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            
            # Draw a green dot on the live frame for feedback
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 2)
            
            cv2.circle(canvas, (cx, cy), 30, (255, 255, 255), -1) #🦊 use to change chickness of drawing

    # --- Displaying Information ---
    # Display prediction text on the frame
    cv2.putText(frame, prediction_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Combine the live frame with the drawing canvas
    combined = cv2.add(frame, canvas)

    # Show the windows
    cv2.imshow("Virtual Drawing", combined)
    cv2.imshow("Mask", mask) # Use this window to help tune your color

    # --- Keyboard Controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Clear the canvas
        canvas = np.zeros_like(frame)
        prediction_text = ""
    elif key == ord('p'):  # Predict the drawn digit
        digit, confidence = predict_drawn_digit(canvas)
        prediction_text = f"Predicted: {digit} ({confidence:.2f}%)"

# === STEP 5: CLEANUP ===
cap.release()
cv2.destroyAllWindows()