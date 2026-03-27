import socket
import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components import containers
import sys
import os
import urllib.request

UDP_IP   = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Download pose model if needed ---
pose_model_path = os.path.join(script_dir, "pose_landmarker.task")
if not os.path.exists(pose_model_path):
    print("[PoseSender] Downloading pose model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        pose_model_path
    )
    print("[PoseSender] Pose model downloaded.")

# --- Download object detection model if needed ---
det_model_path = os.path.join(script_dir, "efficientdet_lite0.tflite")
if not os.path.exists(det_model_path):
    print("[PoseSender] Downloading object detection model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
        det_model_path
    )
    print("[PoseSender] Detection model downloaded.")

# --- Pose landmarker ---
pose_base    = mp_python.BaseOptions(model_asset_path=pose_model_path)
pose_options = mp_vision.PoseLandmarkerOptions(
    base_options=pose_base,
    running_mode=mp_vision.RunningMode.VIDEO
)
pose_detector = mp_vision.PoseLandmarker.create_from_options(pose_options)

# --- Object detector ---
det_base    = mp_python.BaseOptions(model_asset_path=det_model_path)
det_options = mp_vision.ObjectDetectorOptions(
    base_options=det_base,
    running_mode=mp_vision.RunningMode.VIDEO,
    score_threshold=0.3,
    category_allowlist=["sports ball"]
)
obj_detector = mp_vision.ObjectDetector.create_from_options(det_options)

# --- Video source ---
if len(sys.argv) >= 2:
    arg = sys.argv[1]
    try:
        source = int(arg)
    except ValueError:
        source = arg
else:
    source = 0

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"[PoseSender] Could not open: {source}")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

is_video = isinstance(source, str)
print(f"[PoseSender] {'Video file' if is_video else 'Camera'}: {source}")
print("[PoseSender] Running. Press Q to quit.")

# Landmark indices
IDX_WRIST_R    = 16
IDX_WRIST_L    = 15
IDX_SHOULDER_R = 12
IDX_SHOULDER_L = 11
IDX_ANKLE_R    = 28
IDX_ANKLE_L    = 27

paused     = False
frame_idx  = 0

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            if is_video:
                print("[PoseSender] End of video.")
            break

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if timestamp_ms == 0:
        timestamp_ms = frame_idx * 33
    frame_idx += 1

    # --- Pose ---
    pose_results = pose_detector.detect_for_video(mp_image, timestamp_ms)
    pose_data = {"valid": False}

    if pose_results.pose_landmarks:
        lms = pose_results.pose_landmarks[0]
        def px(i):
            return [lms[i].x * w, lms[i].y * h]
        pose_data = {
            "valid":         True,
            "wristRight":    px(IDX_WRIST_R),
            "wristLeft":     px(IDX_WRIST_L),
            "shoulderRight": px(IDX_SHOULDER_R),
            "shoulderLeft":  px(IDX_SHOULDER_L),
            "ankleRight":    px(IDX_ANKLE_R),
            "ankleLeft":     px(IDX_ANKLE_L),
        }
        # Draw skeleton
        for idx in [IDX_WRIST_R, IDX_WRIST_L, IDX_SHOULDER_R,
                    IDX_SHOULDER_L, IDX_ANKLE_R, IDX_ANKLE_L]:
            lm = lms[idx]
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (0,255,0), -1)

    # --- Ball detection ---
    det_results = obj_detector.detect_for_video(mp_image, timestamp_ms)
    ball_data   = None

    for det in det_results.detections:
        bb   = det.bounding_box
        cx   = bb.origin_x + bb.width  / 2
        cy   = bb.origin_y + bb.height / 2
        r    = (bb.width + bb.height)  / 4
        ball_data = [cx, cy, r]
        cv2.rectangle(frame,
            (bb.origin_x, bb.origin_y),
            (bb.origin_x + bb.width, bb.origin_y + bb.height),
            (0, 200, 255), 2)
        cv2.circle(frame, (int(cx), int(cy)), int(r), (0, 255, 200), 2)
        score = det.categories[0].score
        cv2.putText(frame, f"ball {score:.2f}",
            (bb.origin_x, bb.origin_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
        break

    # --- Send UDP ---
    data = {**pose_data, "ball": ball_data}
    sock.sendto(json.dumps(data).encode(), (UDP_IP, UDP_PORT))

    display = cv2.resize(frame, (1280, 720))
    cv2.imshow("Pose + Ball Sender", display)

    key = cv2.waitKey(30 if is_video else 1) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' ') and is_video:
        paused = not paused
        print("[PoseSender]", "Paused" if paused else "Resumed")

cap.release()
cv2.destroyAllWindows()
sock.close()
