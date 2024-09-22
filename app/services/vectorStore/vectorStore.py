
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = Chroma(
    collection_name="lecture_collection",
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

# vector_store = ""


def getVectorStore():
    print(vector_store)
    return vector_store