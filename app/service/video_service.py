
from fastapi import File, UploadFile

def create_upload_video(video: UploadFile = File(...)):
    return {"filename": video.filename}