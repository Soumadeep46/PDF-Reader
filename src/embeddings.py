from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np

def chunk_and_embed_text(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        embedded_chunks = embeddings.embed_documents(chunks)

        return chunks, embedded_chunks
    except Exception as e:
        raise Exception(f"Failed to chunk or embed text: {str(e)}")

def setup_faiss_index(embedded_chunks):
    try:
        vectors = np.array(embedded_chunks).astype('float32')
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        return index
    except Exception as e:
        raise Exception(f"Failed to set up FAISS index: {str(e)}")

def query_faiss_index(index, query_embedding, k=1):
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
    return indices[0]

