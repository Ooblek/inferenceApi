import json
import time
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

from routes import router
origins = [
    "*",
    "http://192.168.1.12:3000",
    "null",
    "http://localhost:1420",
    "https://tauri.localhost"
]
application = FastAPI()
application.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    expose_headers=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
application.include_router(router)
cleaned_lecture = ''


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(application, host="0.0.0.0", port=8000)