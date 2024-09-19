from pydantic import BaseModel

class PrepareRequest(BaseModel):
    videoUrl: str

