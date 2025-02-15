from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.pdf_processor import process_pdf
from src.embeddings import chunk_and_embed_text, setup_faiss_index, query_faiss_index
from src.llm import setup_llm, chat_response, summarize_content
from src.config import Settings

app = FastAPI()
settings = Settings()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        pdf_text = await process_pdf(file)
        chunks, embedded_chunks = chunk_and_embed_text(pdf_text)
        faiss_index = setup_faiss_index(embedded_chunks)
        return {"message": "PDF processed successfully", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    try:
        llm = setup_llm()
        chunks, embedded_chunks = chunk_and_embed_text(settings.pdf_text)
        faiss_index = setup_faiss_index(embedded_chunks)
        similar_chunk_indices = query_faiss_index(faiss_index, request.question)
        context = chunks[similar_chunk_indices[0]]
        response = chat_response(llm, request.question, context)
        return {"response": response, "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize")
async def summarize():
    try:
        summary = summarize_content(settings.pdf_text[:2000])
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

