import re
from .vectorStore import vectorStore
from fastapi.responses import StreamingResponse
import asyncio
from .vectorStore import vectorStore
from langchain_text_splitters import TokenTextSplitter
from sumy.summarizers.text_rank  import TextRankSummarizer as Summarizer
from sumy.utils import get_stop_words
from sumy.nlp.stemmers import Stemmer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import tiktoken


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
        await asyncio.sleep(5)
        yield b"some fake video bytes"

def getSummary(llm, lecture):
    encoding = tiktoken.get_encoding('gpt2')
    num_tokens = len(encoding.encode(lecture))
    num_chunks = 0
    print(num_tokens)
    if(num_tokens > 3900):
       num_chunks = num_tokens/2
    else:
       num_chunks = num_tokens
    text_splitter = TokenTextSplitter(chunk_size=int(num_chunks), chunk_overlap=0)
    texts = text_splitter.split_text(lecture)
   #  Check if the last chunk has data and remove it if it doesn't
    if(len(texts[-1]) < 1024):
       texts.pop()
    print(len(texts))
    if(len(texts) > 1):
       yield "\n\n This lecture was divided into "+str(len(texts))+" Segments. \nPlease wait...\n"
       index = 0
       for text in texts:
          if(len(text) >= 1024):
            index += 1
            yield "\n Segment "+str(index)+": \n"
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
   results = vector_store.similarity_search_with_relevance_scores(requestChat, k=2)
   template = """<|user|>
            You will be given some context. Answer questions only based on that. Do not make up any answers. Do not elaborate on anything that is not found in the context.
            <|end|>
            <|user|>Context: {0}.
            Question: {1}.
            <|end|><|assistant|>"""
   print(results)
   complete_results = ''
   for r in results:
      if(r[1] > 0):
         complete_results += r[0].page_content
         
   print(complete_results)
   session = llm(
            template.format(complete_results, requestChat), # Prompt
            stream=True,
            max_tokens=None, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["<|endoftext|>"], # Stop generating just before the model would generate a new question
            echo=False # Echo the prompt back in the output
      )
   for chunk in session:
    delta = chunk['choices'][0]['text']
    print(delta)
    yield delta


   

   