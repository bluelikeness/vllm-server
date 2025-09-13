import io
from typing import Any
import PyPDF2
import docx


def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_file = io.BytesIO(file_content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"PDF 텍스트 추출 오류: {e}")
        return ""


def extract_text_from_docx(file_content: bytes) -> str:
    try:
        doc_file = io.BytesIO(file_content)
        d = docx.Document(doc_file)
        text = ""
        for paragraph in d.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"DOCX 텍스트 추출 오류: {e}")
        return ""


def process_uploaded_file(file_content: bytes, file_type: str) -> str:
    file_type = file_type.lower()
    if file_type == "pdf":
        return extract_text_from_pdf(file_content)
    elif file_type in ["docx", "doc"]:
        return extract_text_from_docx(file_content)
    elif file_type == "txt":
        return file_content.decode('utf-8', errors='ignore')
    else:
        return f"지원하지 않는 파일 형식: {file_type}"
