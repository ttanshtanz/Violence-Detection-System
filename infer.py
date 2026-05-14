import cv2
import time
import winsound
import numpy as np
import os

from ultralytics import YOLO
from tensorflow.keras.models import load_model

# ======================================================
# SETTINGS
# ======================================================

IMG_SIZE = 64
SEQUENCE_LENGTH = 8
VIOLENCE_THRESHOLD = 0.70

ALARM_COOLDOWN = 5

sequence = []

last_alert_time = 0

# ======================================================
# LOAD MODELS
# ======================================================

print("[INFO] Loading YOLO model...")

yolo = YOLO("yolov8n.pt")

print("[INFO] Loading violence model...")

violence_model = load_model("violence_model.h5")

# ======================================================
# START ViDEO
# ======================================================

VIDEO_PATH = r"E:\Violence Detection\test_videos\violence.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open video file")
    exit()

print("[INFO] Starting detection...")

# ======================================================
# MAIN LOOP
# ======================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ==================================================
    # YOLO PERSON DETECTION
    # ==================================================

    results = yolo(frame, verbose=False)

    persons = []

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])

            # PERSON CLASS
            if cls == 0:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                persons.append((x1, y1, x2, y2))

    # ==================================================
    # PREPARE FRAME FOR VIOLENCE MODEL
    # ==================================================

    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    normalized = (resized / 255.0).astype(np.float32)

    sequence.append(normalized)

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    violence_detected = False
    violence_prob = 0.0

    # ==================================================
    # PREDICT
    # ==================================================

    if len(sequence) == SEQUENCE_LENGTH:

        input_data = np.expand_dims(sequence, axis=0)

        prediction = violence_model.predict(
            input_data,
            verbose=0
        )[0][0]

        violence_prob = float(prediction)

        if violence_prob > VIOLENCE_THRESHOLD:
            violence_detected = True

    # ==================================================
    # DRAW BOXES
    # ==================================================

    for (x1, y1, x2, y2) in persons:

        if violence_detected:
            color = (0,0,255)
        else:
            color = (0,255,0)

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            color,
            2
        )

    # ==================================================
    # DISPLAY STATUS
    # ==================================================

    if violence_detected:

        print(f"[ALERT] Violence detected with probability {violence_prob:.2f}")

        text = f"VIOLENCE DETECTED ({violence_prob:.2f})"

        cv2.putText(
            frame,
            text,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

        # ==============================================
        # PLAY ALARM
        # ==============================================

        current_time = time.time()

        if current_time - last_alert_time > ALARM_COOLDOWN:

            winsound.PlaySound(
                "alarm.wav",
                winsound.SND_FILENAME | winsound.SND_ASYNC
            )
            last_alert_time = current_time

    else:

        text = f"NORMAL ({violence_prob:.2f})"

        cv2.putText(
            frame,
            text,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            3
        )

    # ==================================================
    # SHOW WINDOW
    # ==================================================

    cv2.imshow("Violence Detection System", frame)

    # ==================================================
    # EXIT
    # ==================================================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================================================
# CLEANUP
# ======================================================

cap.release()

cv2.destroyAllWindows()