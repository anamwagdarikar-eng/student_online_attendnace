# Online Attendance System

A Streamlit-based attendance system with:

- Student facial registration using six photos from multiple angles
- Classroom scan from webcam or RTSP CCTV stream
- PostgreSQL storage for student encodings and attendance records
- Attendance dashboard with Excel and PDF exports

## Project structure

- `app.py` – main Streamlit application
- `src/db.py` – PostgreSQL connection and data access layer
- `src/vision.py` – face detection, encoding, and classroom matching logic
- `src/reporting.py` – Excel and PDF exports

## Setup

> Important: the face-recognition stack (including dlib) works best on Python 3.11 or 3.12. The machine in this workspace currently has Python 3.13 only, so the build step for dlib/face-recognition will fail unless you switch to a compatible interpreter.

1. Create a virtual environment with Python 3.11 or 3.12:
   py -3.12 -m venv .venv
   .venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Configure the PostgreSQL connection:
   copy .env.example .env
   Update `DATABASE_URL` if needed.

4. Run the app on a machine connected to the camera's LAN:
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501

   Open `http://localhost:8501` on that machine, or `http://<machine-LAN-IP>:8501` from another device on the same LAN. Allow inbound TCP port 8501 in the machine's firewall if other devices need to access the app.

   Configure `RTSP_HOST`, `RTSP_USERNAME`, and `RTSP_PASSWORD` in a local `.env` file. RTSP connections are made by the Streamlit process, so VLC should be able to open the camera from this same machine.

5. Render deployment start command:
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0

## Multi-camera classroom setup

- The app checks all 16 classroom RTSP URLs automatically by cycling through the camera IDs.
- Each camera is mapped to a classroom such as Class 1 through Class 16.
- The dashboard lists online and offline status for each camera and lets you run a class scan directly from the panel.

## Notes

- Each registered student is stored using six face embeddings captured from different angles.
- The class scan can work with a file upload, browser webcam, or RTSP stream.
- The app stores attendance data and can export a class report as XLS and PDF.
