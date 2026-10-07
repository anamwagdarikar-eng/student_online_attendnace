from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def export_excel_report(records_df: pd.DataFrame):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        records_df.to_excel(writer, index=False, sheet_name="Attendance")
    buffer.seek(0)
    return buffer.getvalue()


def export_pdf_report(records_df: pd.DataFrame):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    title = Paragraph("Attendance Report", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    data = [["Student ID", "Student Name", "Class", "Confidence"]]
    for _, row in records_df.iterrows():
        data.append([
            row.get("student_id", ""),
            row.get("student_name", ""),
            row.get("class_name", ""),
            row.get("confidence", ""),
        ])

    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
