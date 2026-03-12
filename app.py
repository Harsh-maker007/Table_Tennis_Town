import os
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import requests

try:
    import torch  # noqa: F401
except Exception:
    torch = None
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


Point = Tuple[int, int]
Box = Tuple[int, int, int, int]
FACE_MODEL_URL = "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt"


@dataclass
class ShotEvent:
    frame_idx: int
    player_id: str
    contact_point: Point
    end_reason: str
    zone: str


@dataclass
class RallyState:
    in_rally: bool = False
    last_hit_frame: int = -1
    last_hit_player: Optional[str] = None
    hit_count: int = 0
    rally_start_frame: int = -1
    last_seen_frame: int = -1


@dataclass
class PlayerStats:
    player_id: str
    handedness: str
    hits: int = 0
    points: int = 0
    zone_hits: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    zone_wins: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_hit(self, zone: str) -> None:
        self.hits += 1
        self.zone_hits[zone] += 1

    def record_point(self, zone: str) -> None:
        self.points += 1
        self.zone_wins[zone] += 1


class BallTracker:
    def __init__(
        self,
        color: str,
        min_area: int,
        max_area: int,
    ) -> None:
        self.color = color
        self.min_area = min_area
        self.max_area = max_area
        self.trail: Deque[Point] = deque(maxlen=32)

    def _mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if self.color == "orange":
            lower = np.array([5, 120, 120])
            upper = np.array([20, 255, 255])
        else:
            lower = np.array([0, 0, 180])
            upper = np.array([180, 40, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask

    def detect(self, frame: np.ndarray) -> Optional[Point]:
        mask = self._mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        best = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            if area > best_area:
                best_area = area
                best = cnt
        if best is None:
            return None
        (x, y), radius = cv2.minEnclosingCircle(best)
        if radius < 2:
            return None
        center = (int(x), int(y))
        self.trail.append(center)
        return center

    def draw(self, frame: np.ndarray) -> None:
        for i in range(1, len(self.trail)):
            if self.trail[i - 1] is None or self.trail[i] is None:
                continue
            thickness = int(np.sqrt(len(self.trail) / float(i + 1)) * 2)
            cv2.line(frame, self.trail[i - 1], self.trail[i], (0, 255, 255), thickness)


class SimpleKalman:
    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.last_point: Optional[Point] = None

    def predict(self) -> Optional[Point]:
        pred = self.kf.predict()
        x, y = int(pred[0][0]), int(pred[1][0])
        return (x, y)

    def update(self, point: Optional[Point]) -> Optional[Point]:
        if point is None:
            return self.predict()
        meas = np.array([[np.float32(point[0])], [np.float32(point[1])]])
        self.kf.correct(meas)
        self.last_point = point
        return point


def clamp_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


def get_zone(point: Point, table_box: Box) -> str:
    x, y = point
    x1, y1, x2, y2 = table_box
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return "unknown"
    x_norm = (x - x1) / width
    y_norm = (y - y1) / height
    col = "center"
    if x_norm < 0.33:
        col = "left"
    elif x_norm > 0.66:
        col = "right"
    row = "short" if y_norm < 0.5 else "deep"
    return f"{row}-{col}"


def detect_hit(
    positions: Deque[Point],
    frame_idx: int,
    center_line_x: int,
    min_speed: float,
    racket_boxes: Optional[List[Box]],
    racket_distance: int,
    require_racket_contact: bool,
) -> Optional[Tuple[str, Point]]:
    if len(positions) < 4:
        return None
    p1, p2, p3, p4 = positions[-4], positions[-3], positions[-2], positions[-1]
    vx1 = p2[0] - p1[0]
    vx2 = p4[0] - p3[0]
    speed = abs(vx2) + abs(p4[1] - p3[1])
    if speed < min_speed:
        return None
    if vx1 == 0 or vx2 == 0:
        return None
    if np.sign(vx1) == np.sign(vx2):
        return None
    if require_racket_contact:
        if not racket_boxes:
            return None
        nearest_box = None
        nearest_dist = None
        for (x1, y1, x2, y2) in racket_boxes:
            cx = min(max(p4[0], x1), x2)
            cy = min(max(p4[1], y1), y2)
            dist = int(np.hypot(p4[0] - cx, p4[1] - cy))
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_box = (x1, y1, x2, y2)
        if nearest_dist is None or nearest_dist > racket_distance:
            return None
        if nearest_box is not None:
            bx1, _, bx2, _ = nearest_box
            player_id = "A" if (bx1 + bx2) // 2 < center_line_x else "B"
        else:
            player_id = "A" if p4[0] < center_line_x else "B"
    else:
        player_id = "A" if p4[0] < center_line_x else "B"
    return player_id, p4


def update_rally(
    rally: RallyState,
    hit: Optional[Tuple[str, Point]],
    frame_idx: int,
    loss_timeout: int,
) -> Optional[str]:
    if hit is not None:
        player_id, _ = hit
        if not rally.in_rally:
            rally.in_rally = True
            rally.rally_start_frame = frame_idx
            rally.hit_count = 0
        rally.hit_count += 1
        rally.last_hit_frame = frame_idx
        rally.last_hit_player = player_id
        rally.last_seen_frame = frame_idx
        return None
    if rally.in_rally and frame_idx - rally.last_seen_frame > loss_timeout:
        rally.in_rally = False
        return "lost_ball"
    return None


def summarize_player(stats: PlayerStats) -> Dict[str, str]:
    if stats.hits == 0:
        return {"strength": "no data", "weakness": "no data"}
    zone_rates = {}
    for zone, hits in stats.zone_hits.items():
        wins = stats.zone_wins.get(zone, 0)
        zone_rates[zone] = wins / max(1, hits)
    if not zone_rates:
        return {"strength": "no data", "weakness": "no data"}
    strength = max(zone_rates, key=zone_rates.get)
    weakness = min(zone_rates, key=zone_rates.get)
    return {"strength": strength, "weakness": weakness}


def extract_boxes(results, class_id: int, conf_threshold: float) -> List[Box]:
    boxes: List[Box] = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item()) if box.conf is not None else 1.0
            if cls == int(class_id) and conf >= conf_threshold:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                boxes.append((x1, y1, x2, y2))
    return boxes


def crop_box(box: Box, width: int, height: int, pad: int = 0) -> Box:
    x1, y1, x2, y2 = box
    return clamp_box((x1 - pad, y1 - pad, x2 + pad, y2 + pad), width, height)


def ensure_file(path: str, url: str) -> Optional[str]:
    if not path or not url:
        return None
    if os.path.exists(path):
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        return None
    except Exception as exc:
        return str(exc)


def detect_motion_ball(
    prev_gray: Optional[np.ndarray],
    gray: np.ndarray,
    roi_box: Box,
    min_area: int,
    max_area: int,
    predicted: Optional[Point],
) -> Optional[Point]:
    if prev_gray is None:
        return None
    rx1, ry1, rx2, ry2 = roi_box
    prev_roi = prev_gray[ry1:ry2, rx1:rx2]
    curr_roi = gray[ry1:ry2, rx1:rx2]
    if prev_roi.shape != curr_roi.shape:
        return None
    diff = cv2.absdiff(prev_roi, curr_roi)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = None
    best_dist = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < 2:
            continue
        px, py = int(x) + rx1, int(y) + ry1
        if predicted is None:
            return (px, py)
        dist = np.hypot(px - predicted[0], py - predicted[1])
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = (px, py)
    return best


def draw_overlay(
    frame: np.ndarray,
    table_box: Box,
    center_line_x: int,
    rally: RallyState,
    stats_a: PlayerStats,
    stats_b: PlayerStats,
    mode: str,
) -> None:
    x1, y1, x2, y2 = table_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.line(frame, (center_line_x, y1), (center_line_x, y2), (0, 255, 255), 2)
    cv2.putText(
        frame,
        f"Rally: {'ON' if rally.in_rally else 'OFF'}  Hits: {rally.hit_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"A hits: {stats_a.hits} pts: {stats_a.points}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 255),
        2,
    )
    cv2.putText(
        frame,
        f"B hits: {stats_b.hits} pts: {stats_b.points}",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 200, 0),
        2,
    )
    if mode == "training":
        cv2.putText(
            frame,
            "Training: focus on ball path + hand zone",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


def build_stats_panel(stats_a: PlayerStats, stats_b: PlayerStats) -> None:
    col1, col2 = st.columns(2)
    summary_a = summarize_player(stats_a)
    summary_b = summarize_player(stats_b)
    with col1:
        st.subheader("Player A")
        st.write(f"Handedness: {stats_a.handedness}")
        st.write(f"Hits: {stats_a.hits}  Points: {stats_a.points}")
        st.write(f"Strength zone: {summary_a['strength']}")
        st.write(f"Weakness zone: {summary_a['weakness']}")
    with col2:
        st.subheader("Player B")
        st.write(f"Handedness: {stats_b.handedness}")
        st.write(f"Hits: {stats_b.hits}  Points: {stats_b.points}")
        st.write(f"Strength zone: {summary_b['strength']}")
        st.write(f"Weakness zone: {summary_b['weakness']}")


def main() -> None:
    st.set_page_config(page_title="Table Tennis Rally Analyzer", layout="wide")
    st.title("Table Tennis Rally Analyzer")
    st.caption("Match mode: analyze both players. Training mode: focus on improvement + ball path.")

    if torch is None:
        st.warning("PyTorch not found. The app will run in heuristic mode only.")
    if YOLO is None:
        st.warning("Ultralytics YOLOv8 not found. Install ultralytics to enable YOLO modes.")

    mode = st.sidebar.selectbox("Mode", ["match", "training"])
    source = st.sidebar.selectbox("Source", ["Laptop webcam", "Video file", "IP camera URL", "Mobile snapshot"])
    players_in_view = st.sidebar.selectbox("Players in view", ["two", "one"])
    color = st.sidebar.selectbox("Ball color", ["white", "orange"])
    min_area = st.sidebar.slider("Ball min area", 5, 300, 30)
    max_area = st.sidebar.slider("Ball max area", 300, 5000, 1200)
    min_speed = st.sidebar.slider("Hit min speed", 10, 80, 25)
    loss_timeout = st.sidebar.slider("Lost ball frames", 5, 90, 25)
    st.sidebar.markdown("Performance")
    max_fps = st.sidebar.slider("Max FPS", 5, 60, 20)
    process_every = st.sidebar.slider("Process every Nth frame", 1, 5, 1)
    display_every = st.sidebar.slider("Display every Nth frame", 1, 5, 1)
    target_width = st.sidebar.slider("Frame width", 320, 1280, 640, step=40)
    auto_reconnect = st.sidebar.checkbox("Auto-reconnect video", value=True)
    reconnect_delay = st.sidebar.slider("Reconnect delay (ms)", 200, 3000, 800, step=100)
    reconnect_max = st.sidebar.slider("Reconnect attempts", 1, 10, 3)
    st.sidebar.markdown("Detections")
    detect_faces = st.sidebar.checkbox("Detect faces (Haar)", value=False)
    use_yolo_person = st.sidebar.checkbox("Use YOLOv8 for players", value=True)
    yolo_person_model = st.sidebar.text_input("YOLO person model", "yolov8n.pt")
    yolo_person_conf = st.sidebar.slider("YOLO person conf", 10, 80, 25)
    ball_mode = st.sidebar.selectbox("Ball detection mode", ["hybrid", "yolo", "hsv"])
    use_yolo_ball = st.sidebar.checkbox("Use YOLOv8 for ball (sports ball)", value=True)
    yolo_ball_model = st.sidebar.text_input("YOLO ball model", "yolov8n.pt")
    yolo_ball_class = st.sidebar.number_input("Ball class id (COCO sports ball = 32)", 0, 80, 32)
    yolo_ball_conf = st.sidebar.slider("YOLO ball conf", 5, 80, 15)
    yolo_imgsz = st.sidebar.slider("YOLO input size", 320, 1280, 640, step=32)
    yolo_roi_infer = st.sidebar.checkbox("YOLO on table ROI only", value=True)
    yolo_roi_pad = st.sidebar.slider("YOLO ROI pad (px)", 0, 120, 30)
    use_motion_fallback = st.sidebar.checkbox("Use motion fallback for ball", value=True)
    motion_min_area = st.sidebar.slider("Motion min area", 5, 200, 20)
    motion_max_area = st.sidebar.slider("Motion max area", 200, 2000, 800)
    use_yolo_racket = st.sidebar.checkbox("Use YOLOv8 for racket", value=True)
    yolo_racket_class = st.sidebar.number_input("Racket class id (COCO tennis racket = 38)", 0, 80, 38)
    yolo_racket_conf = st.sidebar.slider("YOLO racket conf", 5, 80, 20)
    require_racket_contact = st.sidebar.checkbox("Require racket contact for hit", value=True)
    racket_distance = st.sidebar.slider("Racket contact distance (px)", 10, 150, 60)
    use_yolo_face = st.sidebar.checkbox("Use YOLOv8 for face", value=True)
    yolo_face_model = st.sidebar.text_input("YOLO face model", "models/yolov8n-face-lindevs.pt")
    yolo_face_class = st.sidebar.number_input("Face class id", 0, 80, 0)
    use_yolo_table = st.sidebar.checkbox("Use YOLOv8 for table (dining table)", value=False)
    yolo_table_class = st.sidebar.number_input("Table class id (COCO dining table = 60)", 0, 80, 60)
    yolo_table_conf = st.sidebar.slider("YOLO table conf", 5, 80, 20)
    face_scale = st.sidebar.slider("Face scale", 11, 15, 13)
    face_neighbors = st.sidebar.slider("Face neighbors", 3, 8, 4)
    player_a_hand = st.sidebar.selectbox("Player A handedness", ["right", "left"])
    player_b_hand = st.sidebar.selectbox("Player B handedness", ["right", "left"])

    st.sidebar.markdown("Table ROI (percent of frame)")
    roi_x1 = st.sidebar.slider("ROI left", 0, 100, 15)
    roi_x2 = st.sidebar.slider("ROI right", 0, 100, 85)
    roi_y1 = st.sidebar.slider("ROI top", 0, 100, 20)
    roi_y2 = st.sidebar.slider("ROI bottom", 0, 100, 85)
    st.sidebar.markdown("Player Split (percent of frame)")
    split_x = st.sidebar.slider("Center split", 10, 90, 50)

    frame_placeholder = st.empty()
    stats_placeholder = st.container()

    if "run" not in st.session_state:
        st.session_state.run = False

    start = st.button("Start")
    stop = st.button("Stop")
    if start:
        st.session_state.run = True
    if stop:
        st.session_state.run = False

    if source == "Video file":
        uploaded = st.file_uploader("Upload a match video", type=["mp4", "mov", "avi"])
        loop_video = st.checkbox("Loop video", value=True)
    else:
        uploaded = None
        loop_video = False

    ip_url = None
    if source == "IP camera URL":
        ip_url = st.text_input("IP camera stream URL", "")

    tracker = BallTracker(color=color, min_area=min_area, max_area=max_area)
    kalman = SimpleKalman()
    rally = RallyState()
    stats_a = PlayerStats("A", player_a_hand)
    stats_b = PlayerStats("B", player_b_hand)
    face_cascade = None
    if detect_faces:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
    yolo_person = None
    yolo_ball = None
    yolo_face = None
    if YOLO is not None:
        yolo_models = st.session_state.get("yolo_models", {})

        def load_yolo(path: str) -> Optional["YOLO"]:
            if not path:
                return None
            model = yolo_models.get(path)
            if model is None:
                model = YOLO(path)
                yolo_models[path] = model
                st.session_state["yolo_models"] = yolo_models
            return model

        if use_yolo_person:
            try:
                yolo_person = load_yolo(yolo_person_model)
            except Exception as exc:
                st.warning(f"YOLO person model load failed: {exc}")
                yolo_person = None
        if use_yolo_ball and yolo_ball_model:
            try:
                yolo_ball = load_yolo(yolo_ball_model)
            except Exception as exc:
                st.warning(f"YOLO ball model load failed: {exc}")
                yolo_ball = None
        if use_yolo_face and yolo_face_model:
            try:
                if yolo_face_model.endswith("yolov8n-face-lindevs.pt"):
                    err = ensure_file(yolo_face_model, FACE_MODEL_URL)
                    if err:
                        st.warning(f"Face model download failed: {err}")
                yolo_face = load_yolo(yolo_face_model)
            except Exception as exc:
                st.warning(f"YOLO face model load failed: {exc}")
                yolo_face = None

    if source == "Mobile snapshot":
        shot = st.camera_input("Take a snapshot")
        if shot is not None:
            bytes_data = shot.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                height, width = frame.shape[:2]
                if width > target_width:
                    scale = target_width / float(width)
                    frame = cv2.resize(frame, (target_width, int(height * scale)))
                height, width = frame.shape[:2]
                table_box = clamp_box(
                    (
                        int(width * roi_x1 / 100),
                        int(height * roi_y1 / 100),
                        int(width * roi_x2 / 100),
                        int(height * roi_y2 / 100),
                    ),
                    width,
                    height,
                )
                center_line_x = int(width * split_x / 100)
                ball = None
                ball_results = None
                shared_results = None
                roi_box = table_box
                if yolo_roi_infer:
                    roi_box = crop_box(table_box, width, height, yolo_roi_pad)
                rx1, ry1, rx2, ry2 = roi_box
                roi = frame[ry1:ry2, rx1:rx2]
                if yolo_ball is not None and yolo_person is not None and yolo_ball is yolo_person:
                    shared_conf = min(yolo_ball_conf / 100.0, yolo_person_conf / 100.0)
                    shared_results = yolo_ball.predict(
                        roi if yolo_roi_infer else frame,
                        imgsz=yolo_imgsz,
                        conf=shared_conf,
                        verbose=False,
                    )
                    ball_results = shared_results
                elif yolo_ball is not None:
                    ball_results = yolo_ball.predict(
                        roi if yolo_roi_infer else frame,
                        imgsz=yolo_imgsz,
                        conf=yolo_ball_conf / 100.0,
                        verbose=False,
                    )
                if use_yolo_table and (yolo_person is not None or yolo_ball is not None):
                    table_results = shared_results
                    if table_results is None:
                        model = yolo_person if yolo_person is not None else yolo_ball
                        if model is not None:
                            table_results = model.predict(
                                frame,
                                imgsz=yolo_imgsz,
                                conf=yolo_table_conf / 100.0,
                                verbose=False,
                            )
                    if table_results is not None:
                        tables = extract_boxes(table_results, yolo_table_class, yolo_table_conf / 100.0)
                        if tables:
                            tables = sorted(
                                tables,
                                key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                                reverse=True,
                            )
                            table_box = clamp_box(tables[0], width, height)
                if ball_mode in ("yolo", "hybrid") and ball_results and ball_results[0].boxes is not None:
                    for box in ball_results[0].boxes:
                        cls = int(box.cls[0].item())
                        if cls == int(yolo_ball_class):
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                            if yolo_roi_infer:
                                x1, x2 = x1 + rx1, x2 + rx1
                                y1, y2 = y1 + ry1, y2 + ry1
                            ball = ((x1 + x2) // 2, (y1 + y2) // 2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            break
                if ball is None and ball_mode in ("hsv", "hybrid"):
                    ball = tracker.detect(frame)
                ball = kalman.update(ball)
                tracker.draw(frame)
                racket_boxes: List[Box] = []
                if use_yolo_racket and (yolo_person is not None or yolo_ball is not None):
                    racket_results = shared_results
                    if racket_results is None:
                        model = yolo_person if yolo_person is not None else yolo_ball
                        if model is not None:
                            racket_results = model.predict(
                                frame,
                                imgsz=target_width,
                                conf=yolo_racket_conf / 100.0,
                                verbose=False,
                            )
                    if racket_results is not None:
                        racket_boxes = extract_boxes(
                            racket_results,
                            yolo_racket_class,
                            yolo_racket_conf / 100.0,
                        )
                for (x1, y1, x2, y2) in racket_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                if ball:
                    zone = get_zone(ball, table_box)
                    cv2.circle(frame, ball, 8, (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"Zone: {zone}",
                        (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                if yolo_face is not None:
                    results = yolo_face.predict(
                        roi if yolo_roi_infer else frame,
                        imgsz=yolo_imgsz,
                        conf=0.25,
                        verbose=False,
                    )
                    if results and results[0].boxes is not None:
                        for box in results[0].boxes:
                            cls = int(box.cls[0].item())
                            if cls == int(yolo_face_class):
                                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                                if yolo_roi_infer:
                                    x1, x2 = x1 + rx1, x2 + rx1
                                    y1, y2 = y1 + ry1, y2 + ry1
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                elif face_cascade is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=face_scale / 10.0,
                        minNeighbors=face_neighbors,
                        minSize=(30, 30),
                    )
                    for (fx, fy, fw, fh) in faces:
                        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 200, 0), 2)
                draw_overlay(frame, table_box, center_line_x, rally, stats_a, stats_b, mode)
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        return

    if not st.session_state.run:
        st.info("Press Start to begin processing.")
        return

    def open_capture() -> Optional[cv2.VideoCapture]:
        if source == "Video file":
            if uploaded is None:
                return None
            return cv2.VideoCapture(uploaded.name)
        if source == "IP camera URL":
            if not ip_url:
                return None
            return cv2.VideoCapture(ip_url)
        return cv2.VideoCapture(0)

    cap = open_capture()

    if not cap.isOpened():
        st.error("Failed to open the video source.")
        return

    frame_idx = 0
    shots: List[ShotEvent] = []
    last_frame_time = time.time()
    prev_gray = None

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            if source == "Video file" and loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            elif auto_reconnect:
                cap.release()
                success = False
                for _ in range(reconnect_max):
                    time.sleep(reconnect_delay / 1000.0)
                    cap = open_capture()
                    if cap is not None and cap.isOpened():
                        success = True
                        break
                if not success:
                    st.warning("Video source disconnected.")
                    break
                continue
            else:
                break
        if frame_idx % process_every != 0:
            frame_idx += 1
            continue
        height, width = frame.shape[:2]
        if width > target_width:
            scale = target_width / float(width)
            frame = cv2.resize(frame, (target_width, int(height * scale)))
        height, width = frame.shape[:2]
        table_box = clamp_box(
            (
                int(width * roi_x1 / 100),
                int(height * roi_y1 / 100),
                int(width * roi_x2 / 100),
                int(height * roi_y2 / 100),
            ),
            width,
            height,
        )
        center_line_x = int(width * split_x / 100)
        ball = None
        ball_results = None
        shared_results = None
        roi_box = table_box
        if yolo_roi_infer:
            roi_box = crop_box(table_box, width, height, yolo_roi_pad)
        rx1, ry1, rx2, ry2 = roi_box
        roi = frame[ry1:ry2, rx1:rx2]
        if yolo_ball is not None and yolo_person is not None and yolo_ball is yolo_person:
            shared_conf = min(yolo_ball_conf / 100.0, yolo_person_conf / 100.0)
            shared_results = yolo_ball.predict(
                roi if yolo_roi_infer else frame,
                imgsz=yolo_imgsz,
                conf=shared_conf,
                verbose=False,
            )
            ball_results = shared_results
        elif yolo_ball is not None:
            ball_results = yolo_ball.predict(
                roi if yolo_roi_infer else frame,
                imgsz=yolo_imgsz,
                conf=yolo_ball_conf / 100.0,
                verbose=False,
            )
        if use_yolo_table and (yolo_person is not None or yolo_ball is not None):
            table_results = shared_results
            if table_results is None:
                model = yolo_person if yolo_person is not None else yolo_ball
                if model is not None:
                    table_results = model.predict(
                        frame,
                        imgsz=yolo_imgsz,
                        conf=yolo_table_conf / 100.0,
                        verbose=False,
                    )
            if table_results is not None:
                tables = extract_boxes(table_results, yolo_table_class, yolo_table_conf / 100.0)
                if tables:
                    tables = sorted(
                        tables,
                        key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                        reverse=True,
                    )
                    table_box = clamp_box(tables[0], width, height)
        if ball_mode in ("yolo", "hybrid") and ball_results and ball_results[0].boxes is not None:
            for box in ball_results[0].boxes:
                cls = int(box.cls[0].item())
                if cls == int(yolo_ball_class):
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    if yolo_roi_infer:
                        x1, x2 = x1 + rx1, x2 + rx1
                        y1, y2 = y1 + ry1, y2 + ry1
                    ball = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    break
        if ball is None and ball_mode in ("hsv", "hybrid"):
            ball = tracker.detect(frame)
        if ball is None and use_motion_fallback:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            predicted = kalman.predict() if kalman is not None else None
            ball = detect_motion_ball(
                prev_gray,
                gray,
                roi_box,
                motion_min_area,
                motion_max_area,
                predicted,
            )
            prev_gray = gray
        else:
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ball = kalman.update(ball)
        if ball is not None:
            rally.last_seen_frame = frame_idx
        racket_boxes: List[Box] = []
        if use_yolo_racket and (yolo_person is not None or yolo_ball is not None):
            racket_results = shared_results
            if racket_results is None:
                model = yolo_person if yolo_person is not None else yolo_ball
                if model is not None:
                    racket_results = model.predict(
                        roi if yolo_roi_infer else frame,
                        imgsz=yolo_imgsz,
                        conf=yolo_racket_conf / 100.0,
                        verbose=False,
                    )
            if racket_results is not None:
                racket_boxes = extract_boxes(
                    racket_results,
                    yolo_racket_class,
                    yolo_racket_conf / 100.0,
                )
                if yolo_roi_infer:
                    racket_boxes = [
                        (x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1) for (x1, y1, x2, y2) in racket_boxes
                    ]
        hit = detect_hit(
            tracker.trail,
            frame_idx,
            center_line_x,
            min_speed,
            racket_boxes,
            racket_distance,
            require_racket_contact,
        )
        if players_in_view == "one" and hit:
            hit = ("A", hit[1])
        if hit:
            player_id, contact = hit
            zone = get_zone(contact, table_box)
            if player_id == "A":
                stats_a.record_hit(zone)
            else:
                stats_b.record_hit(zone)
            shots.append(
                ShotEvent(
                    frame_idx=frame_idx,
                    player_id=player_id,
                    contact_point=contact,
                    end_reason="hit",
                    zone=zone,
                )
            )
        end_reason = update_rally(rally, hit, frame_idx, loss_timeout)
        if end_reason == "lost_ball" and rally.last_hit_player:
            last_zone = shots[-1].zone if shots else "unknown"
            if rally.last_hit_player == "A":
                stats_a.record_point(last_zone)
            else:
                stats_b.record_point(last_zone)

        if ball is not None:
            cv2.circle(frame, ball, 6, (0, 0, 255), 2)
        tracker.draw(frame)
        for (x1, y1, x2, y2) in racket_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if yolo_face is not None:
            results = yolo_face.predict(
                roi if yolo_roi_infer else frame,
                imgsz=yolo_imgsz,
                conf=0.25,
                verbose=False,
            )
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0].item())
                    if cls == int(yolo_face_class):
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        if yolo_roi_infer:
                            x1, x2 = x1 + rx1, x2 + rx1
                            y1, y2 = y1 + ry1, y2 + ry1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        elif face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=face_scale / 10.0,
                minNeighbors=face_neighbors,
                minSize=(30, 30),
            )
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 200, 0), 2)
        if yolo_person is not None and players_in_view == "two":
            results = shared_results
            if results is None:
                results = yolo_person.predict(
                    roi if yolo_roi_infer else frame,
                    imgsz=yolo_imgsz,
                    conf=yolo_person_conf / 100.0,
                    verbose=False,
                )
            if results and results[0].boxes is not None:
                boxes = []
                for box in results[0].boxes:
                    cls = int(box.cls[0].item())
                    if cls == 0:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        if yolo_roi_infer:
                            x1, x2 = x1 + rx1, x2 + rx1
                            y1, y2 = y1 + ry1, y2 + ry1
                        boxes.append((x1, y1, x2, y2))
                boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:2]
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
        draw_overlay(frame, table_box, center_line_x, rally, stats_a, stats_b, mode)

        if frame_idx % display_every == 0:
            frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            with stats_placeholder:
                if players_in_view == "one":
                    build_stats_panel(stats_a, PlayerStats("B", player_b_hand))
                else:
                    build_stats_panel(stats_a, stats_b)

        frame_idx += 1
        if max_fps > 0:
            elapsed = time.time() - last_frame_time
            sleep_for = max(0.0, (1.0 / max_fps) - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_frame_time = time.time()

    cap.release()


if __name__ == "__main__":
    main()
