from youtube_transcript_api import YouTubeTranscriptApi
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from .vectorStore import vectorStore
from langchain_core.documents import Document

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
filler_words = ['uh', 'um', 'basically', 'so', 'hmm', 'okay']
def getTranscript(videoId):
    vector_store = vectorStore.getVectorStore()
    transcripts = YouTubeTranscriptApi.get_transcript(videoId)
    lecture = ''
    if(transcripts):
        #join into a string 
        for t in transcripts:
            text = t['text'].lower()
            text = text.replace(u'\xa0', u' ')
            text = text.replace(u'\n', u' ')
            text_tokens = word_tokenize(text)
            cleaned_text = ''
            for token in text_tokens:
                if(token not in filler_words and token not in stop_words):
                    lemma = wnl.lemmatize(token)
                    cleaned_text = cleaned_text+ lemma + ' '
            lecture += cleaned_text
    lecture_texts = text_splitter.create_documents([lecture])
    uuids = [str(uuid4()) for _ in range(len(lecture_texts))]
    vector_store.reset_collection()
    vector_store.add_documents(documents=lecture_texts, ids=uuids)
    return lecture
