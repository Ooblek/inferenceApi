from typing import List
from fastapi import APIRouter, HTTPException, status, File
from typing_extensions import Annotated
from models.chatRequest import ChatRequest
from models.prepareRequest import PrepareRequest, PrepareSrt
from services import prepareVideo, summarizer
from llama_cpp import Llama
from fastapi.responses import StreamingResponse


router = APIRouter(tags=["inference"])
llm = Llama(
    model_path="E:\proj\diss\inferenceApi\\app\slm\Phi-3.5-Q6_K_L.gguf",
    n_ctx=5000,
    n_batch=224,
    n_threads=8,
    top_p=1,
    n_gpu_layers=30
)
# llm = "test"
template = """<|user|>
            You will be given a lecture. Answer only based on that. Do not make up any answers
            <|end|>
            <|user|>lecture: {}.
            Task: Give me a summary of this lecture covering everything that was taught.
            <|end|><|assistant|>"""

lecture = ''
indexed_lecture = ''
plain_lecture = {}

@router.post('/uploadSrt')
async def prepareSrt(request: List[PrepareSrt]):
    transcriptDict = request
    transcripts = prepareVideo.prepareVideo(transcriptDict)
    global indexed_lecture
    indexed_lecture = transcripts['indexed']
    global lecture
    lecture = transcripts['transcripts']
    global plain_lecture
    plain_lecture = transcripts['plain_transcripts']
    return transcripts

@router.post("/prepare")
async def prepareData(request: PrepareRequest):
    requestDict = request.model_dump()
    print(requestDict['videoUrl'])
    transcripts = prepareVideo.getTranscript(requestDict['videoUrl'])
    print(transcripts)
    global indexed_lecture
    indexed_lecture = transcripts['indexed']
    global lecture
    lecture = transcripts['transcripts']
    global plain_lecture
    plain_lecture = transcripts['plain_transcripts']
    return transcripts

@router.get('/importantRegions')
async def getRegions():
    global lecture
    global plain_lecture
    global indexed_lecture
    return summarizer.getRegions(indexed_lecture, len(plain_lecture),plain_lecture )

@router.get('/summarize')
def getSummary():
    global lecture    
    global llm
    if(lecture):
        # return StreamingResponse(summarizer.fake_video_streamer(), media_type='text/event-stream')
        return  StreamingResponse(summarizer.getSummary(llm, lecture), media_type='text/event-stream')
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Lecture not found. Has it been loaded?"
    )

@router.post('/chat')
async def chatWithLecture(request: ChatRequest):
    requestDict = request.model_dump()
    print(requestDict)
    # return summarizer.getChat(llm, requestDict['searchString'])
    # return  StreamingResponse(summarizer.fake_video_streamer(), media_type='text/event-stream')
    return  StreamingResponse(summarizer.getChat(llm, requestDict['searchString']), media_type='text/event-stream')


