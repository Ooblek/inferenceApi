from fastapi import APIRouter

from .inferenceRoute import router as inference_router

router = APIRouter(prefix="/v1")

router.include_router(inference_router)
