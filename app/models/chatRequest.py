from pydantic import BaseModel

class ChatRequest(BaseModel):
    searchString: str

