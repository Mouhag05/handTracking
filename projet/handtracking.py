import cv2
import numpy as nppip

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
      break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hand = max(contours, key=cv2.contourArea)
    cv2.drawContours(frame, [hand], -1, (0, 255, 0), 2)
    cv2.imshow('Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
