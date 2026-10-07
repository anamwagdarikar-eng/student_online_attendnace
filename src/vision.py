import os
from typing import Dict, List, Tuple

import cv2
import face_recognition
import numpy as np
from PIL import Image


DEFAULT_MATCH_THRESHOLD = 0.45


def load_image_from_upload(uploaded_file):
    """Load a PIL-compatible image from Streamlit file uploader."""
    image = Image.open(uploaded_file)
    return np.array(image.convert("RGB"))


def extract_face_embedding(image_array):
    """Return the first face encoding from a single-person image."""
    rgb = cv2.cvtColor(np.array(image_array), cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    if not face_locations:
        raise ValueError("No face detected in the uploaded photo. Please use a clearer image.")

    face_encodings = face_recognition.face_encodings(rgb, face_locations)
    if not face_encodings:
        raise ValueError("No valid facial encoding could be generated from the uploaded photo.")

    return np.array(face_encodings[0], dtype=float).tolist()


def find_best_student(student_templates, candidate_encoding):
    """Compare a candidate encoding against all registered student templates."""
    best_match = None
    best_distance = 1.0

    for student_id, data in student_templates.items():
        for embedding in data["embeddings"]:
            candidate = np.asarray(embedding, dtype=float)
            current_dist = face_recognition.face_distance([candidate], np.asarray(candidate_encoding, dtype=float))[0]
            if current_dist < best_distance:
                best_distance = float(current_dist)
                best_match = {"student_id": student_id, "student_name": data["student_name"], "class_name": data["class_name"], "distance": best_distance}

    if best_match is None:
        return None, 1.0

    return best_match, best_distance


def process_class_image(image_array, student_templates):
    """Detect each face in a class image and match it against registered students."""
    rgb_image = cv2.cvtColor(np.array(image_array), cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

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


def open_rtsp_frame(rtsp_url: str):
    """Open a frame from an RTSP stream for class scanning."""
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {rtsp_url}")

    ret, frame = cap.read()
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
