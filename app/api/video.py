
from fastapi import APIRouter, File, UploadFile

from service import video_service

router = APIRouter()

@router.get("/")
async def read_root():
    return {"Hello": "Universe"}

@router.post("/uploadvideo/")
async def create_upload_video(video: UploadFile = File(...)):
    return video_service.create_upload_video(video)