import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    TimeDistributed,
    GlobalAveragePooling2D,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# SETTINGS
# https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset downloaded using kaggle API and extracted to the path below
# =========================================================
DATASET_PATH = r"E:\Violence Detection\real-life-violence-situations-dataset\Real Life Violence Dataset"
IMG_SIZE = 64
SEQUENCE_LENGTH = 8

X = []
y = []

# =========================================================
# EXTRACT FRAMES FROM VIDEO
# =========================================================
def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(total_frames // SEQUENCE_LENGTH, 1)
    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = (frame / 255.0).astype(np.float32)
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.float32)

# =========================================================
# LOAD DATASET
# =========================================================
classes = ["NonViolence", "Violence"]

print("\n[INFO] Loading dataset...\n")
for label, class_name in enumerate(classes):
    class_path = os.path.join(DATASET_PATH, class_name)
    videos = os.listdir(class_path)
    videos = videos[:200]
    print(f"[INFO] Processing {class_name} videos...")
    for video_name in tqdm(videos):
        video_path = os.path.join(class_path, video_name)
        frames = extract_frames(video_path)
        if len(frames) == SEQUENCE_LENGTH:
            X.append(frames)
            y.append(label)

# =========================================================
# CONVERT TO NUMPY
# =========================================================
X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\n[INFO] Dataset Loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# =========================================================
# BUILD MODEL
# =========================================================
print("\n[INFO] Building model...\n")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

model = Sequential([
    TimeDistributed(
        base_model,
        input_shape=(SEQUENCE_LENGTH,224,224,3)
    ),
    TimeDistributed(GlobalAveragePooling2D()),
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# =========================================================
# CALLBACKS
# =========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =========================================================
# TRAIN MODEL
# =========================================================

print("\n[INFO] Training started...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=2,
    callbacks=[early_stop]
)

# =========================================================
# EVALUATE
# =========================================================

loss, accuracy = model.evaluate(X_test, y_test)

print("\n====================================")
print(f"TEST ACCURACY: {accuracy*100:.2f}%")
print("====================================")

# =========================================================
# SAVE MODEL
# =========================================================

model.save("violence_model.h5")

print("\n[INFO] Model saved as violence_model.h5")