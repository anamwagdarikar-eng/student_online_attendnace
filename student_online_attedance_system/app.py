import os
from urllib.parse import quote

import cv2
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.db import (
    create_schema,
    get_attendance_records,
    get_registered_students,
    save_attendance_session,
    save_student_registration,
)
from src.reporting import export_excel_report, export_pdf_report
from src.vision import (
    extract_face_embedding,
    get_camera_capture,
    load_image_from_upload,
    open_rtsp_frame,
    process_class_image,
)

load_dotenv()

st.set_page_config(page_title="Online Attendance System", layout="wide")

CAMERA_COUNT = int(os.getenv("CAMERA_COUNT", "16"))
RTSP_HOST = os.getenv("RTSP_HOST", "192.168.100.127")
RTSP_USERNAME = os.getenv("RTSP_USERNAME", "admin")
RTSP_PASSWORD = quote(os.getenv("RTSP_PASSWORD", "Vinu@2710"), safe="")

DEFAULT_CLASS_NAMES = [f"Class {index}" for index in range(1, CAMERA_COUNT + 1)]


def generate_rtsp_url(camera_id: int) -> str:
    return (
        f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_HOST}:554/cam/realmonitor?"
        f"channel={camera_id}&subtype=0"
    )


def auto_detect_rtsp_cameras() -> dict:
    detected = {}
    for camera_id in range(1, CAMERA_COUNT + 1):
        rtsp_url = generate_rtsp_url(camera_id)
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            detected[camera_id] = {"status": "offline", "url": rtsp_url, "class_name": f"Class {camera_id}"}
            continue

        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None and frame.size > 0:
            detected[camera_id] = {"status": "online", "url": rtsp_url, "class_name": f"Class {camera_id}"}
        else:
            detected[camera_id] = {"status": "offline", "url": rtsp_url, "class_name": f"Class {camera_id}"}

    return detected


if "attendance_rows" not in st.session_state:
    st.session_state.attendance_rows = []
if "camera_status" not in st.session_state:
    st.session_state.camera_status = {}

create_schema()

st.title("Online Attendance System")

sidebar = st.sidebar
sidebar.header("Classroom network")
if sidebar.button("Auto-detect 16 RTSP cameras"):
    st.session_state.camera_status = auto_detect_rtsp_cameras()

if st.session_state.camera_status:
    detected_candidates = [
        item["class_name"] for item in st.session_state.camera_status.values() if item["status"] == "online"
    ]
else:
    detected_candidates = DEFAULT_CLASS_NAMES

TAB_REGISTRATION, TAB_SCAN, TAB_REPORT = "Registration", "Class Scan", "Attendance Report"
selected_tab = st.tabs([TAB_REGISTRATION, TAB_SCAN, TAB_REPORT])

with selected_tab[0]:
    st.subheader("Student Photo Registration")
    with st.form("student_registration_form"):
        student_id = st.text_input("Student ID")
        student_name = st.text_input("Student Name")
        class_name = st.selectbox("Class / Section", options=DEFAULT_CLASS_NAMES, index=0)

        st.write("Upload six photos from different angles for the same student.")
        uploaded_images = []
        for idx in range(1, 7):
            uploaded = st.file_uploader(f"Photo {idx} (angle {idx})", type=["png", "jpg", "jpeg"], key=f"photo_{idx}")
            uploaded_images.append(uploaded)

        submit_registration = st.form_submit_button("Register Student")

    if submit_registration:
        valid_uploads = [img for img in uploaded_images if img is not None]

        if not student_id or not student_name or not class_name:
            st.warning("Please complete the student details before registering.")
        elif len(valid_uploads) != 6:
            st.warning("Please upload exactly six photos for registration.")
        else:
            try:
                embeddings = []
                for image_file in valid_uploads:
                    image_array = load_image_from_upload(image_file)
                    embedding = extract_face_embedding(image_array)
                    embeddings.append(embedding)

                save_student_registration(student_id, student_name, class_name, embeddings)
                st.success(f"Student {student_name} registered successfully.")
            except Exception as exc:
                st.error(f"Registration failed: {exc}")

with selected_tab[1]:
    st.subheader("Multi-camera classroom dashboard")

    if st.session_state.camera_status:
        camera_columns = st.columns(4)
        for index, (camera_id, camera_data) in enumerate(st.session_state.camera_status.items()):
            with camera_columns[index % 4]:
                st.markdown(f"### {camera_data['class_name']}")
                st.caption(f"Camera ID: {camera_id}")
                st.caption(f"URL: {camera_data['url']}")
                if camera_data["status"] == "online":
                    st.success("RTSP camera online")
                else:
                    st.warning("Camera offline or unreachable")

                if st.button(f"Scan {camera_data['class_name']}", key=f"scan_{camera_id}"):
                    class_name = camera_data["class_name"]
                    rtsp_url = camera_data["url"]
                    try:
                        class_image = open_rtsp_frame(rtsp_url)
                        registered_students = get_registered_students(class_name)
                        if not registered_students:
                            st.warning(f"No students registered for {class_name} yet.")
                        else:
                            recognized_results, annotated_frame = process_class_image(class_image, registered_students)
                            st.image(annotated_frame, channels="RGB")
                            st.session_state.attendance_rows = recognized_results
                            if recognized_results:
                                st.dataframe(pd.DataFrame(recognized_results))
                            else:
                                st.info("No matching faces were found in the classroom feed.")
                    except Exception as exc:
                        st.error(f"Unable to scan {class_name}: {exc}")
    else:
        class_name = st.selectbox("Select class", options=DEFAULT_CLASS_NAMES, index=0)
        source = st.radio("Capture source", ["Web Camera", "RTSP Camera", "Upload Class Photo"])

        class_image = None
        if source == "Web Camera":
            st.info("This uses the local camera attached to the machine running the app.")
            try:
                class_image = get_camera_capture()
                st.image(class_image, channels="RGB")
            except Exception as exc:
                st.warning(f"Web camera not available: {exc}")

        if source == "RTSP Camera":
            rtsp_url = st.text_input(
                "RTSP URL",
                value=os.getenv("RTSP_URL", generate_rtsp_url(2)),
                help="Use the classroom CCTV feed or a local RTSP camera stream.",
            )
            if st.button("Load RTSP Frame"):
                try:
                    class_image = open_rtsp_frame(rtsp_url)
                    st.image(class_image, channels="RGB")
                except Exception as exc:
                    st.error(f"RTSP error: {exc}")

        if source == "Upload Class Photo":
            uploaded_class_photo = st.file_uploader("Upload class photo", type=["png", "jpg", "jpeg"])
            if uploaded_class_photo is not None:
                class_image = load_image_from_upload(uploaded_class_photo)
                st.image(class_image, channels="RGB")

        if class_image is not None:
            if st.button("Scan Class"):
                registered_students = get_registered_students(class_name)
                if not registered_students:
                    st.warning("No students are registered for this class yet.")
                else:
                    recognized_results, annotated_frame = process_class_image(class_image, registered_students)
                    st.image(annotated_frame, channels="RGB")
                    st.session_state.attendance_rows = recognized_results

                    if recognized_results:
                        st.dataframe(pd.DataFrame(recognized_results))
                    else:
                        st.info("No known faces matched in this class photo.")

    if st.session_state.attendance_rows:
        if st.button("Submit Attendance"):
            class_name = st.selectbox("Submit for class", options=DEFAULT_CLASS_NAMES, index=0, key="submit_class")
            session_id = save_attendance_session(class_name, "class_scan", st.session_state.attendance_rows)
            if session_id:
                st.success(f"Attendance submitted for {len(st.session_state.attendance_rows)} recognized students.")
            else:
                st.warning("No attendance records to submit.")

with selected_tab[2]:
    st.subheader("Attendance Dashboard")
    report_class = st.selectbox("Select class to report", options=DEFAULT_CLASS_NAMES, index=0)

    attendance_rows = get_attendance_records(report_class)
    if attendance_rows:
        df = pd.DataFrame(attendance_rows)
        st.dataframe(df)

        excel_bytes = export_excel_report(df)
        pdf_bytes = export_pdf_report(df)

        st.download_button(
            label="Download XLS",
            data=excel_bytes,
            file_name=f"attendance_{report_class.lower().replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=f"attendance_{report_class.lower().replace(' ', '_')}.pdf",
            mime="application/pdf",
        )
    else:
        st.info("No attendance records found for this class yet.")
