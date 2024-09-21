from youtube_transcript_api import YouTubeTranscriptApi
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from .vectorStore import vectorStore
from langchain_core.documents import Document
import re
from langchain_text_splitters import TokenTextSplitter



text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=0)

wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
filler_words = ['uh', 'um', 'basically', 'so', 'hmm', 'okay']
def getTranscript(videoId):
    vector_store = vectorStore.getVectorStore()
    transcripts = YouTubeTranscriptApi.get_transcript(videoId)

    total_duration = 0
    index = 0
    complete_transcript = ''
    complete_transcript_indexed = ''
    if(transcripts):
        #join into a string 
        for t in transcripts:
            total_duration += t['duration']
            text = t['text']
            text = text.replace('\n', ' ')
            text = text.replace("\n", " ").strip("...").replace(".", "").replace("?", " ").replace("!", " ")
            text = re.sub(r"\[.+?\]", '', text)
            complete_transcript += text +'. '
            complete_transcript_indexed += text+' ' + str(index)+' . '
            index += 1
    texts = text_splitter.split_text(complete_transcript)
    vector_store = vectorStore.getVectorStore()
    vector_store.reset_collection()
    documents = []
    for text in texts:
        document = Document(
            page_content=text,
            id=uuid4()
        )
        documents.append(document)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    return complete_transcript_indexed

