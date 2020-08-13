
from fastapi import APIRouter

from models.item_model import Payload
from service import item_service

router = APIRouter()


@router.get("/")
async def read_root():
    return {"Hello": "Universe"}

@router.post("/indexitem")
async def index_item(payload: Payload):
    return item_service.index_item(payload)