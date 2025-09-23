from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import shutil

from app.settings import STORAGE_DIR

router = APIRouter(prefix="/files", tags=["Files"])

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    dst = STORAGE_DIR / file.filename
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename}

@router.get("/download/{filename}")
async def download(filename: str):
    p = STORAGE_DIR / filename
    if not p.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(p, media_type="application/octet-stream", filename=p.name)
