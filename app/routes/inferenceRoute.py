from fastapi import APIRouter, HTTPException, status
from models.chatRequest import ChatRequest
from models.prepareRequest import PrepareRequest
from services import prepareVideo, summarizer
from llama_cpp import Llama
from fastapi.responses import StreamingResponse


router = APIRouter(tags=["inference"])
llm = Llama(
    model_path="E:\proj\diss\inferenceApi\\app\slm\Phi-3.5-Q6_K_L.gguf",
    n_ctx=4096,
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

@router.post("/prepare")
async def prepareData(request: PrepareRequest):
    requestDict = request.model_dump()
    print(requestDict['videoUrl'])
    transcripts = prepareVideo.getTranscript(requestDict['videoUrl'])
    global lecture
    lecture = transcripts
    return transcripts

@router.get('/summarize')
async def getSummary():
    global lecture    
    global llm
    if(lecture):
        return  StreamingResponse(summarizer.getSummary(llm, lecture), media_type='text/event-stream')
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Lecture not found. Has it been loaded?"
    )

@router.post('/chat')
async def chatWithLecture(request: ChatRequest):
    requestDict = request.model_dump()
    # return summarizer.getChat(llm, requestDict['searchString'])
    return  StreamingResponse(summarizer.getChat(llm, requestDict['searchString']), media_type='text/event-stream')


