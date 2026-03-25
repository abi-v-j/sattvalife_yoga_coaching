from pathlib import Path
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


# =========================================================
# GLOBAL SMOOTHING STATE
# =========================================================
PRED_HISTORY = deque(maxlen=8)
SCORE_HISTORY = deque(maxlen=8)
FEEDBACK_HISTORY = deque(maxlen=6)

LEFT_KNEE_HISTORY = deque(maxlen=5)
RIGHT_KNEE_HISTORY = deque(maxlen=5)
LEFT_ELBOW_HISTORY = deque(maxlen=5)
RIGHT_ELBOW_HISTORY = deque(maxlen=5)

PERFECT_HOLD_COUNT = 0


def smooth_label(new_label):
    PRED_HISTORY.append(new_label)
    return Counter(PRED_HISTORY).most_common(1)[0][0]


def smooth_score(new_score):
    SCORE_HISTORY.append(float(new_score))
    return int(sum(SCORE_HISTORY) / len(SCORE_HISTORY))


def smooth_feedback(new_feedback):
    FEEDBACK_HISTORY.append(new_feedback)
    return Counter(FEEDBACK_HISTORY).most_common(1)[0][0]


def smooth_value(history, value):
    history.append(float(value))
    return float(sum(history) / len(history))


# =========================================================
# PATHS / MODEL LOAD
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "ml_models" / "model.pkl"
SCALER_PATH = BASE_DIR / "ml_models" / "scaler.pkl"
LABEL_PATH = BASE_DIR / "ml_models" / "label.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "ml_models" / "feature_names.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35,
)

CONFIDENCE_THRESHOLD = 0.45


# =========================================================
# LANDMARK INDEXES
# =========================================================
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

SELECTED_POINTS = [
    NOSE,
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE
]

POINT_NAME_MAP = {
    NOSE: "nose",
    LEFT_SHOULDER: "left_shoulder",
    RIGHT_SHOULDER: "right_shoulder",
    LEFT_ELBOW: "left_elbow",
    RIGHT_ELBOW: "right_elbow",
    LEFT_WRIST: "left_wrist",
    RIGHT_WRIST: "right_wrist",
    LEFT_HIP: "left_hip",
    RIGHT_HIP: "right_hip",
    LEFT_KNEE: "left_knee",
    RIGHT_KNEE: "right_knee",
    LEFT_ANKLE: "left_ankle",
    RIGHT_ANKLE: "right_ankle",
}

POSE_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
]

GREEN = "#00ff66"
RED = "#ff3b30"
YELLOW = "#ffd60a"
GRAY = "#cfcfcf"
WHITE = "#ffffff"


# =========================================================
# BASIC VIEWS
# =========================================================
def camera_page(request):
    return render(request, "User/camera.html")


def HomePage(request):
    return render(request, "User/home_page.html")


# =========================================================
# RESPONSE HELPERS
# =========================================================
def api_success(**kwargs):
    return JsonResponse({
        "success": True,
        **kwargs
    })


def api_error(message, status=400):
    return JsonResponse({
        "success": False,
        "error": str(message)
    }, status=status)


# =========================================================
# MATH HELPERS
# =========================================================
def calculate_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_angle = np.dot(ba, bc) / denom
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    hip_center = (pts[LEFT_HIP] + pts[RIGHT_HIP]) / 2.0
    pts = pts - hip_center

    shoulder_dist = np.linalg.norm(pts[LEFT_SHOULDER] - pts[RIGHT_SHOULDER])
    if shoulder_dist < 1e-6:
        shoulder_dist = 1.0

    pts = pts / shoulder_dist
    return pts


# =========================================================
# IMAGE HELPERS
# =========================================================
def read_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def enhance_frame(frame):
    frame = cv2.resize(frame, (960, 720))
    # softer than your harsher earlier version
    frame = cv2.convertScaleAbs(frame, alpha=1.10, beta=10)
    return frame


def detect_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None
    return results.pose_landmarks.landmark


def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    return brightness < 45, brightness


def full_body_visible_soft(landmarks):
    try:
        req = {
            "nose": landmarks[NOSE].visibility,
            "left_shoulder": landmarks[LEFT_SHOULDER].visibility,
            "right_shoulder": landmarks[RIGHT_SHOULDER].visibility,
            "left_hip": landmarks[LEFT_HIP].visibility,
            "right_hip": landmarks[RIGHT_HIP].visibility,
            "left_knee": landmarks[LEFT_KNEE].visibility,
            "right_knee": landmarks[RIGHT_KNEE].visibility,
            "left_ankle": landmarks[LEFT_ANKLE].visibility,
            "right_ankle": landmarks[RIGHT_ANKLE].visibility,
        }

        core_ok = (
            req["nose"] > 0.30 and
            req["left_shoulder"] > 0.30 and
            req["right_shoulder"] > 0.30 and
            req["left_hip"] > 0.25 and
            req["right_hip"] > 0.25
        )

        knee_ok = req["left_knee"] > 0.20 or req["right_knee"] > 0.20
        ankle_ok = req["left_ankle"] > 0.15 or req["right_ankle"] > 0.15

        return core_ok and knee_ok and ankle_ok
    except Exception:
        return False


# =========================================================
# MODEL FEATURE CREATION (SELECTED POINTS MODEL)
# 13 landmarks x 3 = 39
# + 10 engineered features = 49
# =========================================================
def create_model_features(landmarks):
    pts = normalize_landmarks(landmarks)

    features = []

    for idx in SELECTED_POINTS:
        features.extend(pts[idx].tolist())

    left_knee = calculate_angle(pts[LEFT_HIP], pts[LEFT_KNEE], pts[LEFT_ANKLE])
    right_knee = calculate_angle(pts[RIGHT_HIP], pts[RIGHT_KNEE], pts[RIGHT_ANKLE])

    left_elbow = calculate_angle(pts[LEFT_SHOULDER], pts[LEFT_ELBOW], pts[LEFT_WRIST])
    right_elbow = calculate_angle(pts[RIGHT_SHOULDER], pts[RIGHT_ELBOW], pts[RIGHT_WRIST])

    left_hip = calculate_angle(pts[LEFT_SHOULDER], pts[LEFT_HIP], pts[LEFT_KNEE])
    right_hip = calculate_angle(pts[RIGHT_SHOULDER], pts[RIGHT_HIP], pts[RIGHT_KNEE])

    shoulder_diff = abs(pts[LEFT_SHOULDER][1] - pts[RIGHT_SHOULDER][1])
    hip_diff = abs(pts[LEFT_HIP][1] - pts[RIGHT_HIP][1])

    left_hand_above_head = float(pts[LEFT_WRIST][1] < pts[NOSE][1])
    right_hand_above_head = float(pts[RIGHT_WRIST][1] < pts[NOSE][1])

    features.extend([
        left_knee, right_knee,
        left_elbow, right_elbow,
        left_hip, right_hip,
        shoulder_diff, hip_diff,
        left_hand_above_head, right_hand_above_head
    ])

    return features, pts


def predict_pose_label(features):
    if len(features) != len(feature_names):
        raise ValueError(f"Feature mismatch: got {len(features)}, expected {len(feature_names)}")

    X_input = pd.DataFrame([features], columns=feature_names)
    X_scaled = scaler.transform(X_input)

    prediction = model.predict(X_scaled)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    confidence = 0.50
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        confidence = float(np.max(probs))

    return str(predicted_label), confidence


# =========================================================
# TREE ANALYSIS
# =========================================================
def analyze_tree_pose(raw_pts):
    ls, rs = raw_pts[LEFT_SHOULDER], raw_pts[RIGHT_SHOULDER]
    le, re = raw_pts[LEFT_ELBOW], raw_pts[RIGHT_ELBOW]
    lw, rw = raw_pts[LEFT_WRIST], raw_pts[RIGHT_WRIST]
    lh, rh = raw_pts[LEFT_HIP], raw_pts[RIGHT_HIP]
    lk, rk = raw_pts[LEFT_KNEE], raw_pts[RIGHT_KNEE]
    la, ra = raw_pts[LEFT_ANKLE], raw_pts[RIGHT_ANKLE]
    nose = raw_pts[NOSE]

    left_knee = smooth_value(LEFT_KNEE_HISTORY, calculate_angle(lh, lk, la))
    right_knee = smooth_value(RIGHT_KNEE_HISTORY, calculate_angle(rh, rk, ra))
    left_elbow = smooth_value(LEFT_ELBOW_HISTORY, calculate_angle(ls, le, lw))
    right_elbow = smooth_value(RIGHT_ELBOW_HISTORY, calculate_angle(rs, re, rw))
    left_hip = calculate_angle(ls, lh, lk)
    right_hip = calculate_angle(rs, rh, rk)

    shoulder_diff = abs(ls[1] - rs[1])
    hip_diff = abs(lh[1] - rh[1])

    shoulder_center = (ls + rs) / 2.0
    hip_center = (lh + rh) / 2.0
    vertical = hip_center + np.array([0, -0.2, 0], dtype=np.float32)
    torso_tilt = calculate_angle(shoulder_center[:2], hip_center[:2], vertical[:2])

    if left_knee > right_knee:
        standing_side = "left"
        stand_knee_idx = LEFT_KNEE
        bent_knee_idx = RIGHT_KNEE
        stand_ankle_idx = LEFT_ANKLE
        bent_ankle_idx = RIGHT_ANKLE
        stand_angle = left_knee
        bent_angle = right_knee
        raised_foot = ra
        stand_hip = lh
        stand_knee = lk
        raised_knee = rk
    else:
        standing_side = "right"
        stand_knee_idx = RIGHT_KNEE
        bent_knee_idx = LEFT_KNEE
        stand_ankle_idx = RIGHT_ANKLE
        bent_ankle_idx = LEFT_ANKLE
        stand_angle = right_knee
        bent_angle = left_knee
        raised_foot = la
        stand_hip = rh
        stand_knee = rk
        raised_knee = lk

    hands_up = (lw[1] < nose[1] + 0.03) and (rw[1] < nose[1] + 0.03)
    arms_straight = left_elbow >= 140 and right_elbow >= 140
    shoulders_level = shoulder_diff < 0.12
    hips_level = hip_diff < 0.12
    torso_ok = torso_tilt < 18

    hand_gap = np.linalg.norm(lw[:2] - rw[:2])
    hand_align_ok = hand_gap < 0.28

    dist_knee = np.linalg.norm(raised_foot[:2] - stand_knee[:2])
    dist_hip = np.linalg.norm(raised_foot[:2] - stand_hip[:2])
    foot_place_ok = (dist_knee < 0.32) or (dist_hip < 0.30)

    center_x = hip_center[0]
    if standing_side == "left":
        knee_open_ok = raised_knee[0] > center_x + 0.005
    else:
        knee_open_ok = raised_knee[0] < center_x - 0.005

    checks = {
        "standing_leg": stand_angle >= 145,
        "bent_leg": 25 <= bent_angle <= 130,
        "foot_place": foot_place_ok,
        "knee_open": knee_open_ok,
        "hands_up": hands_up,
        "arms_straight": arms_straight,
        "hand_align": hand_align_ok,
        "torso": torso_ok,
        "shoulders": shoulders_level,
        "hips": hips_level,
    }

    weights = {
        "standing_leg": 16,
        "bent_leg": 12,
        "foot_place": 14,
        "knee_open": 12,
        "hands_up": 10,
        "arms_straight": 8,
        "hand_align": 4,
        "torso": 10,
        "shoulders": 7,
        "hips": 7,
    }

    score = 0
    for key, ok in checks.items():
        if ok:
            score += weights[key]
    score = min(100, score)

    priority_feedback = []
    if not checks["knee_open"]:
        priority_feedback.append("Open the bent knee outward")
    if not checks["foot_place"]:
        priority_feedback.append("Place the raised foot higher on the inner leg")
    if not checks["standing_leg"]:
        priority_feedback.append("Straighten the standing leg")
    if not checks["hands_up"]:
        priority_feedback.append("Raise both hands above your head")
    if not checks["arms_straight"]:
        priority_feedback.append("Straighten both arms")
    if not checks["hand_align"]:
        priority_feedback.append("Align both hands")
    if not checks["torso"]:
        priority_feedback.append("Keep torso upright")
    if not checks["shoulders"]:
        priority_feedback.append("Level your shoulders")
    if not checks["hips"]:
        priority_feedback.append("Level your hips")

    if score >= 90:
        main_feedback = "Perfect Tree pose"
        status = "perfect"
        pose_label = "Perfect Tree"
    elif score >= 72:
        main_feedback = priority_feedback[0] if priority_feedback else "Very good Tree pose"
        status = "good"
        pose_label = "Tree Pose"
    elif score >= 45:
        main_feedback = priority_feedback[0] if priority_feedback else "You are close. Make small corrections."
        status = "warning"
        pose_label = "Tree Needs Correction"
    else:
        main_feedback = priority_feedback[0] if priority_feedback else "Stand in Tree pose"
        status = "warning"
        pose_label = "Not Ready Yet"

    return {
        "pose_label": pose_label,
        "score": score,
        "status": status,
        "main_feedback": main_feedback,
        "tips": priority_feedback[:3],
        "standing_side": standing_side,
        "stand_knee_idx": stand_knee_idx,
        "bent_knee_idx": bent_knee_idx,
        "stand_ankle_idx": stand_ankle_idx,
        "bent_ankle_idx": bent_ankle_idx,
        "angles": {
            "left_knee_angle": round(left_knee, 2),
            "right_knee_angle": round(right_knee, 2),
            "left_elbow_angle": round(left_elbow, 2),
            "right_elbow_angle": round(right_elbow, 2),
            "left_hip_angle": round(left_hip, 2),
            "right_hip_angle": round(right_hip, 2),
            "torso_tilt": round(torso_tilt, 2),
        },
        "stand_angle": stand_angle,
        "bent_angle": bent_angle,
        "hands_up": hands_up,
        "checks": checks,
    }


def is_tree_like(predicted_label, confidence, stand_angle, bent_angle, hands_up):
    label = str(predicted_label).lower()

    if "tree" in label:
        return True

    if confidence >= 0.55 and stand_angle > 145 and bent_angle < 130:
        return True

    if stand_angle > 150 and bent_angle < 120 and hands_up:
        return True

    return False


# =========================================================
# FRONTEND OVERLAY
# =========================================================
def build_points_for_frontend(raw_pts, analysis, image_width, image_height):
    checks = analysis["checks"]
    stand_knee_idx = analysis["stand_knee_idx"]
    bent_knee_idx = analysis["bent_knee_idx"]
    stand_ankle_idx = analysis["stand_ankle_idx"]
    bent_ankle_idx = analysis["bent_ankle_idx"]

    joint_colors = {idx: GRAY for idx in range(len(raw_pts))}

    for idx in SELECTED_POINTS:
        joint_colors[idx] = GREEN

    joint_colors[stand_knee_idx] = GREEN if checks["standing_leg"] else RED
    joint_colors[bent_knee_idx] = GREEN if (checks["bent_leg"] and checks["knee_open"]) else RED
    joint_colors[stand_ankle_idx] = GREEN if checks["standing_leg"] else RED
    joint_colors[bent_ankle_idx] = GREEN if checks["foot_place"] else RED

    joint_colors[LEFT_ELBOW] = GREEN if checks["arms_straight"] else RED
    joint_colors[RIGHT_ELBOW] = GREEN if checks["arms_straight"] else RED
    joint_colors[LEFT_WRIST] = GREEN if checks["hands_up"] else RED
    joint_colors[RIGHT_WRIST] = GREEN if checks["hands_up"] else RED
    joint_colors[LEFT_SHOULDER] = GREEN if checks["shoulders"] else RED
    joint_colors[RIGHT_SHOULDER] = GREEN if checks["shoulders"] else RED
    joint_colors[LEFT_HIP] = GREEN if checks["hips"] else RED
    joint_colors[RIGHT_HIP] = GREEN if checks["hips"] else RED

    points = []
    for idx, p in enumerate(raw_pts):
        x = float(p[0] * image_width)
        y = float(p[1] * image_height)

        points.append({
            "name": POINT_NAME_MAP.get(idx, f"point_{idx}"),
            "x": x,
            "y": y,
            "color": joint_colors.get(idx, GRAY),
            "visible": True
        })

    return points


def build_lines_for_frontend(analysis):
    checks = analysis["checks"]

    line_colors = []
    for p1, p2 in POSE_CONNECTIONS:
        color = GREEN

        if {p1, p2} & {LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST}:
            if not checks["arms_straight"] or not checks["hands_up"]:
                color = RED

        if {p1, p2} & {LEFT_HIP, RIGHT_HIP}:
            if not checks["hips"]:
                color = RED

        if {p1, p2} & {LEFT_SHOULDER, RIGHT_SHOULDER}:
            if not checks["shoulders"]:
                color = RED

        if {p1, p2} & {LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE}:
            if not (checks["standing_leg"] and checks["bent_leg"]):
                color = RED

        line_colors.append({
            "from": p1,
            "to": p2,
            "color": color
        })

    return line_colors


def build_angle_texts(raw_pts, analysis, image_width, image_height):
    return [
        {
            "text": str(int(analysis["angles"]["left_knee_angle"])),
            "x": float(raw_pts[LEFT_KNEE][0] * image_width),
            "y": float(raw_pts[LEFT_KNEE][1] * image_height),
            "color": YELLOW
        },
        {
            "text": str(int(analysis["angles"]["right_knee_angle"])),
            "x": float(raw_pts[RIGHT_KNEE][0] * image_width),
            "y": float(raw_pts[RIGHT_KNEE][1] * image_height),
            "color": YELLOW
        },
        {
            "text": str(int(analysis["angles"]["left_elbow_angle"])),
            "x": float(raw_pts[LEFT_ELBOW][0] * image_width),
            "y": float(raw_pts[LEFT_ELBOW][1] * image_height),
            "color": YELLOW
        },
        {
            "text": str(int(analysis["angles"]["right_elbow_angle"])),
            "x": float(raw_pts[RIGHT_ELBOW][0] * image_width),
            "y": float(raw_pts[RIGHT_ELBOW][1] * image_height),
            "color": YELLOW
        },
    ]


# =========================================================
# MAIN API
# =========================================================
@csrf_exempt
def predict_yoga_pose(request):
    global PERFECT_HOLD_COUNT

    if request.method != "POST":
        return api_error("Only POST method allowed", status=405)

    if "image" not in request.FILES:
        return api_error("No image uploaded", status=400)

    try:
        uploaded_file = request.FILES["image"]
        frame = read_uploaded_image(uploaded_file)

        if frame is None:
            return api_error("Invalid image file", status=400)

        frame = enhance_frame(frame)

        low_light, brightness = check_lighting(frame)
        if low_light:
            PERFECT_HOLD_COUNT = 0
            return api_success(
                pose="Low Light",
                model_pose="Unknown",
                feedback="Room lighting is too low. Increase light for accurate pose detection.",
                status="warning",
                confidence=0.0,
                score=0,
                angles={},
                details=[
                    "Increase room lighting",
                    "Face the light source",
                    "Avoid dark background"
                ],
                coach_text=f"Low lighting detected ({brightness:.1f}). Improve lighting.",
                perfect_hold=False,
                points=[],
                lines=[],
                angle_texts=[]
            )

        landmarks = detect_landmarks(frame)
        if not landmarks:
            PERFECT_HOLD_COUNT = 0
            return api_success(
                pose="Unknown",
                model_pose="Unknown",
                feedback="No human pose detected. Move back slightly and show full body clearly.",
                status="unknown",
                confidence=0.0,
                score=0,
                angles={},
                details=["Show full body", "Improve room lighting"],
                coach_text="Stand where your full body is visible.",
                perfect_hold=False,
                points=[],
                lines=[],
                angle_texts=[]
            )

        body_visible = full_body_visible_soft(landmarks)

        features, _ = create_model_features(landmarks)
        predicted_label, confidence = predict_pose_label(features)
        stable_predicted_label = smooth_label(predicted_label)

        raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        analysis = analyze_tree_pose(raw_pts)

        if not body_visible:
            analysis["status"] = "warning"
            analysis["main_feedback"] = "Body visibility is low. Move slightly back and keep full body centered."
            analysis["tips"] = [
                "Show full body more clearly",
                "Keep one standing foot fully visible",
                "Stay centered in the frame"
            ]
            analysis["score"] = min(analysis["score"], 45)

        stable_score = smooth_score(analysis["score"])
        stable_feedback = smooth_feedback(analysis["main_feedback"])
        stable_status = analysis["status"]

        tree_like = is_tree_like(
            stable_predicted_label,
            confidence,
            analysis["stand_angle"],
            analysis["bent_angle"],
            analysis["hands_up"]
        )

        if tree_like and stable_score >= 90:
            PERFECT_HOLD_COUNT += 1
        else:
            PERFECT_HOLD_COUNT = 0

        if confidence < CONFIDENCE_THRESHOLD and stable_score < 50:
            return api_success(
                pose="Unknown",
                model_pose=stable_predicted_label,
                feedback="Pose not clear. Stand fully visible in front of the camera.",
                status="unknown",
                confidence=round(float(confidence), 3),
                score=0,
                angles={},
                details=[
                    "Show full body",
                    "Face the camera",
                    "Hold Tree pose steadily"
                ],
                coach_text="Try again with clearer body position.",
                perfect_hold=False,
                points=[],
                lines=[],
                angle_texts=[]
            )

        if not tree_like and stable_score < 40:
            return api_success(
                pose="Not Tree Pose",
                model_pose=stable_predicted_label,
                feedback="Stand in Tree pose",
                status="warning",
                confidence=round(float(confidence), 3),
                score=0,
                angles={},
                details=[
                    "Bring one foot to the inner leg",
                    "Raise both hands above head",
                    "Keep balance on one standing leg"
                ],
                coach_text="Move into Tree pose and hold steady.",
                perfect_hold=False,
                points=[],
                lines=[],
                angle_texts=[]
            )

        pose_name = analysis["pose_label"]
        coach_text = "Stand in Tree pose and improve alignment."

        if stable_score >= 90:
            pose_name = "Perfect Tree"
            coach_text = "Excellent. Hold the pose steadily."
        elif stable_score >= 70:
            pose_name = "Tree Pose"
            coach_text = "Good alignment. Refine the remaining correction."
        elif stable_score >= 45:
            pose_name = "Tree Needs Correction"
            coach_text = "You are close. Make small corrections."

        if PERFECT_HOLD_COUNT >= 3:
            pose_name = "Perfect Tree"
            coach_text = "Hold steady. Excellent balance."

        image_height, image_width = frame.shape[:2]
        points = build_points_for_frontend(raw_pts, analysis, image_width, image_height)
        lines = build_lines_for_frontend(analysis)
        angle_texts = build_angle_texts(raw_pts, analysis, image_width, image_height)

        return api_success(
            pose=pose_name,
            model_pose=stable_predicted_label,
            feedback=stable_feedback,
            status=stable_status,
            confidence=round(float(confidence), 3),
            score=stable_score,
            angles=analysis["angles"],
            details=analysis["tips"],
            coach_text=coach_text,
            perfect_hold=PERFECT_HOLD_COUNT >= 3,
            points=points,
            lines=lines,
            angle_texts=angle_texts
        )

    except Exception as e:
        print("predict_yoga_pose error:", str(e))
        return api_error(str(e), status=500)



        