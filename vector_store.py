from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def get_vector_store() -> Chroma:
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    return Chroma(
        collection_name="trip_reco",
        embedding_function=embedding,
        persist_directory="./chroma_trip"
    )