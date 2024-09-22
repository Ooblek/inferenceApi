import re
from .vectorStore import vectorStore
from fastapi.responses import StreamingResponse
import asyncio
from .vectorStore import vectorStore
from langchain_text_splitters import TokenTextSplitter
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.utils import get_stop_words
from sumy.nlp.stemmers import Stemmer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer



def output_streamer(llm, template):
    session = llm(
            template, # Prompt
            stream=True,
            max_tokens=None, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["<|endoftext|>"], # Stop generating just before the model would generate a new question
            echo=False # Echo the prompt back in the output
      )
    for chunk in session:
      delta = chunk['choices'][0]['text']
      print(delta)
      yield delta

async def fake_video_streamer():
    for i in range(10):
        await asyncio.sleep(10)
        yield b"some fake video bytes"

def getSummary(llm, lecture):
    text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=0)
    texts = text_splitter.split_text(lecture)
    print(len(texts))
    if(len(texts) > 1):
       index = 0
       for text in texts:
          index += 1
          template = """<|user|>You will be given a part of a lecture. This is part {} of {}. Answer only based on that. do not make up any answers. Do not write conclusions unless this is part {}<|end|><|user|>lecture: {}. Task: Summarize this lecture part covering everything that was taught so far.<|end|><|assistant|>"""
          print(template.format(index, len(texts),len(texts), text))
          session = llm(
            template.format(index, len(texts),len(texts), text), # Prompt
            stream=True,
            max_tokens=None, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["<|endoftext|>"], # Stop generating just before the model would generate a new question
            echo=False # Echo the prompt back in the output
          )
          for chunk in session:
            delta = chunk['choices'][0]['text']
            print(delta)
            yield delta
    else:
       template = """<|user|>
            You will be given a lecture. Answer only based on that. Do not make up any answers
            <|end|>
            <|user|>lecture: {}.
            Task: Give me a summary of this lecture covering everything that was taught.
            <|end|><|assistant|>"""
       session = llm(
            template.format(lecture), # Prompt
            stream=True,
            max_tokens=None, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["<|endoftext|>"], # Stop generating just before the model would generate a new question
            echo=False # Echo the prompt back in the output
        )
       for chunk in session:
        delta = chunk['choices'][0]['text']
        print(delta)
        yield delta
       
    
def getRegions(indexed_lecture, total_number, plain_lecture):
   stemmer = Stemmer('english')
   summarizer = Summarizer(stemmer)
   summarizer.stop_words = get_stop_words('english')
   parser = PlaintextParser.from_string(indexed_lecture, Tokenizer('english'))
   summary = []
   for sentence in summarizer(parser.document, int(total_number)/2):
      index = int(re.findall("\(([0-9]+)\)", str(sentence))[0])
      summary.append(plain_lecture[index])
   return summary

def getChat(llm, requestChat):
   vector_store = vectorStore.getVectorStore()
   results = vector_store.similarity_search(requestChat)
   template = """<|user|>
            You will be given some context. Answer questions only based on that. Do not make up any answers. 
            <|end|>
            <|user|>Context: {0}.
            Question: {1}.
            <|end|><|assistant|>"""
   re = results[0].page_content
   session = llm(
            template.format(re, requestChat), # Prompt
            stream=True,
            max_tokens=None, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["<|endoftext|>"], # Stop generating just before the model would generate a new question
            echo=False # Echo the prompt back in the output
      )
   for chunk in session:
    delta = chunk['choices'][0]['text']
    print(delta)
    yield delta


   

   