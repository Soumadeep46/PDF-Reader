import fitz
from fastapi import UploadFile

async def process_pdf(file: UploadFile):
    try:
        pdf_content = await file.read()
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        raise Exception(f"Failed to extract text: {str(e)}")

