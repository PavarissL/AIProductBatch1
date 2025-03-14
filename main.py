from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import qa_chain

app = FastAPI()

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(q: Question):
    result = qa_chain.invoke(q.query)
    return {
        "answer": result['result'],
        "sources": [doc.metadata for doc in result['source_documents']]
    }
