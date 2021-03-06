from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import APIKeyHeader

from api import prediction
from api import healthcheck

api_router = APIRouter()

router = APIRouter()

API_KEY_SCHEME = APIKeyHeader(name='x-api-key')

async def verify_api_key(api_key: str = Depends(API_KEY_SCHEME)):
    if api_key != "dd74decc-8825-4a49-b9bc-e4608249d612":
        raise HTTPException(status_code=400, detail="x-api-key header invalid")
    return api_key

api_router.include_router(
    prediction.router,
    prefix="/prediction",
    tags=["prediction"],
    # dependencies=[Depends(verify_api_key)],
)

api_router.include_router(
    healthcheck.router,
    tags=["healthcheck"],
)
