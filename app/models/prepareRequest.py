from pydantic import BaseModel

class PrepareRequest(BaseModel):
    videoUrl: str

class PrepareSrt(BaseModel):
    text: str
    start: float
    duration: float
    def __getitem__(self, item):
        return getattr(self, item)


