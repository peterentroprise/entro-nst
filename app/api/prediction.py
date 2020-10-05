
from fastapi import APIRouter, File, UploadFile

from service.prediction_service import do_something

router = APIRouter()

@router.get("/")
async def read_root():
    return {"Hello": "Universe"}

@router.get("/nst")
async def nst(request):
    return do_something(request)