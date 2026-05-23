from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

def get_vector_store() -> Chroma:
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    return Chroma(
        collection_name="trip_reco",
        embedding_function=embedding,
        persist_directory="./chroma_trip"
    )
    
def get_agent_vector_store() -> Chroma:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    return Chroma(
        collection_name="trip_reco_agent",
        embedding_function=embedding,
        persist_directory="./chroma_trip_agent"
    )
    
def get_ollama_vector_store() -> Chroma:
    embedding = OllamaEmbeddings(model="qwen2.5:7b")
    
    return Chroma(
        collection_name="trip_reco_ollama",
        embedding_function=embedding,
        persist_directory="./chroma_trip_ollama"
    )