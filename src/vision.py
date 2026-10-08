import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import face_recognition
except ImportError:  # pragma: no cover
    face_recognition = None


DEFAULT_MATCH_THRESHOLD = 0.45


def _to_rgb(image_array):
    image = np.asarray(image_array)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[-1] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.shape[-1] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _scale_and_normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _detect_faces(image_array):
    rgb = _to_rgb(image_array)

    if face_recognition is not None:
        face_locations = face_recognition.face_locations(rgb)
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb, face_locations)
            return face_locations, face_encodings

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detections = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_locations = []
    face_encodings = []

    for (x, y, w, h) in detections:
        face_locations.append((y, x + w, y + h, x))
        face_roi = gray[y : y + h, x : x + w]
        face_roi = cv2.resize(face_roi, (128, 128))
        encoding = _scale_and_normalize(face_roi.flatten())
        face_encodings.append(encoding)

    return face_locations, face_encodings


def load_image_from_upload(uploaded_file):
    """Load a PIL-compatible image from Streamlit file uploader."""
    image = Image.open(uploaded_file)
    return np.array(image.convert("RGB"))


def extract_face_embedding(image_array):
    """Return the first face encoding from a single-person image."""
    face_locations, face_encodings = _detect_faces(image_array)
    if not face_locations or not face_encodings:
        raise ValueError("No face detected in the uploaded photo. Please use a clearer image.")

    embedding = np.asarray(face_encodings[0], dtype=float).tolist()
    return embedding


def find_best_student(student_templates, candidate_encoding):
    """Compare a candidate encoding against all registered student templates."""
    best_match = None
    best_distance = float("inf")
    candidate = np.asarray(candidate_encoding, dtype=float)

    for student_id, data in student_templates.items():
        for embedding in data["embeddings"]:
            template = np.asarray(embedding, dtype=float)
            distance = float(np.linalg.norm(candidate - template))
            if distance < best_distance:
                best_distance = distance
                best_match = {
                    "student_id": student_id,
                    "student_name": data["student_name"],
                    "class_name": data["class_name"],
                    "distance": best_distance,
                }

    if best_match is None:
        return None, 1.0

    return best_match, best_distance


def process_class_image(image_array, student_templates):
    """Detect each face in a class image and match it against registered students."""
    rgb_image = _to_rgb(image_array)
    face_locations, face_encodings = _detect_faces(rgb_image)

    recognized_results = []
    annotated = np.array(image_array.copy())

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        match, distance = find_best_student(student_templates, encoding)
        is_known = match is not None and distance <= DEFAULT_MATCH_THRESHOLD

        label = "Unknown"
        color = (0, 0, 255)
        if is_known:
            label = f"{match['student_name']} ({match['student_id']})"
            color = (0, 255, 0)

        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        cv2.putText(
            annotated,
            label,
            (left, max(top - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        if is_known:
            recognized_results.append(
                {
                    "student_id": match["student_id"],
                    "student_name": match["student_name"],
                    "class_name": match["class_name"],
                    "confidence": round(1 - distance, 4),
                }
            )

    return recognized_results, annotated


def open_rtsp_capture(rtsp_url: str):
    """Open an RTSP stream over TCP using OpenCV's FFmpeg backend."""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    return cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)


def open_rtsp_frame(rtsp_url: str):
    """Open a frame from an RTSP stream for class scanning."""
    cap = open_rtsp_capture(rtsp_url)
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open RTSP stream. Check network access and camera credentials.")

        ret, frame = cap.read()
    finally:
        cap.release()
    if not ret:
        raise RuntimeError("RTSP stream did not return a valid frame.")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def get_camera_capture():
    """Return the first camera frame if available."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No webcam detected on this machine.")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("The webcam could not capture a frame.")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
