from fastapi import APIRouter, FastAPI, File, UploadFile
import os
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = Path("data_warehouse")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/ask")
async def upload_file(file: UploadFile = File(...)):
    file_extension = file.filename.split('.')[-1]
    if file_extension not in ["pdf", "docx"]:
        return {"error": "File type not allowed. Only .pdf and .docx are supported."}

    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"message": f"File {file.filename} uploaded successfully!"}
