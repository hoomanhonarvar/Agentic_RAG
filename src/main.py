from fastapi import FastAPI
from AgentChat import Agentic_RAG
import asyncio

app = FastAPI(title="AutoGen Agent Backend")



@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(query: str):
    retrieve_document,thought,answer = asyncio.run(Agentic_RAG(query))
    print( "retrieve_document : " ,retrieve_document,"\n","thouts: ",thought,"\n","answer: ",answer)
    return {"retrieve_document": retrieve_document, "thought": thought,"answer ": answer}