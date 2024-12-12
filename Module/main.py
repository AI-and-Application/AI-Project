from fastapi import FastAPI, Request
from app.api import analyze_request
from app.models import MyLitModel
from app.transcriber import transcribe
from app.emotion_analysis import analyze_emotion
from app.langchain_prompts import classify_chain, answer_chain
from app.google_search import get_google_results

# FastAPI Initialization
app = FastAPI(
    title="Cha-Vis",
    version="1.0",
    description="AI Navigation Cha-Vis using FastAPI, Whisper, HuBERT, and LangChain."
)

@app.post("/analyze")
async def analyze(request: Request):
    return await analyze_request(request, classify_chain, answer_chain, transcribe, analyze_emotion, get_google_results)
