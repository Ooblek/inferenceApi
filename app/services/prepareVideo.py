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
filler_words = ['uh', 'um', 'basically', 'so', 'hmm', 'okay', 'ok', '--']


def getTranscript(videoId):
    transcripts = YouTubeTranscriptApi.get_transcript(videoId)
    return prepareVideo(transcripts)
    
def prepareVideo(transcripts):
    vector_store = vectorStore.getVectorStore()
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
            words = word_tokenize(text)
            final_sentence = ''
            for word in words:
                if(word not in stop_words and word not in filler_words):
                    final_sentence += wnl.lemmatize(word) + ' '
            complete_transcript += final_sentence +'. '
            complete_transcript_indexed += text+' ' + '(' +str(index)+') . '
            index += 1
    texts = text_splitter.split_text(complete_transcript_indexed)
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
    return {"indexed": complete_transcript_indexed, "transcripts": complete_transcript, "plain_transcripts": transcripts}

