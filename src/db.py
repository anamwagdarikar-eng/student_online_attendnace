import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_bfBoRKZxTt12@ep-spring-voice-b5jxxpf2-pooler.c-7.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def create_schema():
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    student_name TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    department TEXT,
                    camera_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    embedding_id SERIAL PRIMARY KEY,
                    student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
                    angle_index INT NOT NULL,
                    embedding DOUBLE PRECISION[] NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS attendance_sessions (
                    session_id SERIAL PRIMARY KEY,
                    class_name TEXT NOT NULL,
                    submission_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS attendance_records (
                    record_id SERIAL PRIMARY KEY,
                    session_id INT NOT NULL REFERENCES attendance_sessions(session_id) ON DELETE CASCADE,
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    student_name TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    present BOOLEAN NOT NULL DEFAULT TRUE,
                    confidence NUMERIC(5,4),
                    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
        )


def save_student_registration(
    student_id: str,
    student_name: str,
    class_name: str,
    department: str,
    camera_used: str,
    embeddings: List[List[float]],
):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO students (student_id, student_name, class_name, department, camera_used)
                VALUES (:student_id, :student_name, :class_name, :department, :camera_used)
                ON CONFLICT (student_id)
                DO UPDATE SET student_name = EXCLUDED.student_name, class_name = EXCLUDED.class_name, department = EXCLUDED.department, camera_used = EXCLUDED.camera_used
                """
            ),
            {
                "student_id": student_id,
                "student_name": student_name,
                "class_name": class_name,
                "department": department,
                "camera_used": camera_used,
            },
        )

        conn.execute(
            text("DELETE FROM face_embeddings WHERE student_id = :student_id"),
            {"student_id": student_id},
        )

        for index, embedding in enumerate(embeddings):
            conn.execute(
                text(
                    """
                    INSERT INTO face_embeddings (student_id, angle_index, embedding)
                    VALUES (:student_id, :angle_index, :embedding)
                    """
                ),
                {"student_id": student_id, "angle_index": index + 1, "embedding": list(map(float, embedding))},
            )


def get_registered_students(class_name: str | None = None):
    query = "SELECT s.student_id, s.student_name, s.class_name, f.embedding FROM students s LEFT JOIN face_embeddings f ON f.student_id = s.student_id"
    params: Dict[str, Any] = {}

    if class_name:
        query += " WHERE s.class_name = :class_name"
        params["class_name"] = class_name

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    student_templates = {}
    for row in rows:
        student_id = row["student_id"]
        if student_id not in student_templates:
            student_templates[student_id] = {
                "student_name": row["student_name"],
                "class_name": row["class_name"],
                "embeddings": [],
            }
        if row["embedding"] is not None:
            student_templates[student_id]["embeddings"].append(list(row["embedding"]))

    return student_templates


def save_attendance_session(class_name: str, submission_type: str, recognized_students: list):
    if not recognized_students:
        return None

    with engine.begin() as conn:
        session_result = conn.execute(
            text(
                "INSERT INTO attendance_sessions (class_name, submission_type) VALUES (:class_name, :submission_type) RETURNING session_id"
            ),
            {"class_name": class_name, "submission_type": submission_type},
        )
        session_id = session_result.fetchone()[0]

        for student in recognized_students:
            conn.execute(
                text(
                    """
                    INSERT INTO attendance_records (session_id, student_id, student_name, class_name, present, confidence)
                    VALUES (:session_id, :student_id, :student_name, :class_name, TRUE, :confidence)
                    """
                ),
                {
                    "session_id": session_id,
                    "student_id": student["student_id"],
                    "student_name": student["student_name"],
                    "class_name": class_name,
                    "confidence": student.get("confidence", 0.0),
                },
            )

        return session_id


def get_attendance_records(class_name: str | None = None):
    query = "SELECT * FROM attendance_records"
    params: Dict[str, Any] = {}

    if class_name:
        query += " WHERE class_name = :class_name"
        params["class_name"] = class_name

    query += " ORDER BY captured_at DESC"

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    return [dict(row) for row in rows]
