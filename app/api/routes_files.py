from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import shutil

from app import settings

router = APIRouter(prefix="/files", tags=["Files"])


def _contained_path(name: str) -> Path:
    """Join name under STORAGE_DIR and refuse anything escaping it."""
    p = (settings.STORAGE_DIR / name).resolve()
    if p != settings.STORAGE_DIR and settings.STORAGE_DIR not in p.parents:
        raise HTTPException(400, "Path escapes storage directory")
    return p


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Only bare filenames are accepted: no separators, no "..", no absolute paths.
    if not file.filename or Path(file.filename).name != file.filename:
        raise HTTPException(400, "Invalid filename")
    dst = _contained_path(file.filename)
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": file.filename}

@router.get("/download/{filename}")
async def download(filename: str):
    p = STORAGE_DIR / filename
    if not p.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(p, media_type="application/octet-stream", filename=p.name)
