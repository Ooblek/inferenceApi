from fastapi import APIRouter, HTTPException, status
from models.prepareRequest import PrepareRequest
from services import prepareVideo, summarizer

router = APIRouter(tags=["inference"])

lecture = ''

@router.post("/prepare")
async def prepareData(request: PrepareRequest):
    requestDict = request.model_dump()
    print(requestDict['videoUrl'])
    transcripts = prepareVideo.getTranscript(requestDict['videoUrl'])
    global lecture
    lecture = transcripts
    return transcripts

@router.get('/documents')
async def getDocuments():
    global lecture
    if(lecture):
        docs = summarizer.getValuesFromVDB(lecture)
        return docs
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Lecture not found. Has it been loaded?"
    )
    
