import cv2
import numpy as np
from ultralytics import YOLO
import winsound
import time  

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Camera not found")
    exit()

print("Camera is open")

model = YOLO("./best.pt")

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

last_sound_time = 0        
sound_cooldown = 5         

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    frame_diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

    motion_pixels = np.sum(thresh)

    if motion_pixels > 50000:
        results = model.predict(frame, verbose=False)

        label = "unknown"
        confidence = 0.0

        if results and len(results[0].boxes.cls) > 0:
            boxes = results[0].boxes
            names = model.names
            top_idx = boxes.conf.argmax().item()

            
            if time.time() - last_sound_time > sound_cooldown:
                winsound.PlaySound("./alert_tone.wav", winsound.SND_ASYNC)
                last_sound_time = time.time()

            label = names[int(boxes.cls[top_idx])]
            confidence = float(boxes.conf[top_idx])

        cv2.putText(
            frame,
            f"DISTURBANCE: {label.upper()} ({confidence:.2f})",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Smart Farming Vision System", frame)
    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
