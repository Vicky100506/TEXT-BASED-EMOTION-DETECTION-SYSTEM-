from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from prediction import predict   # ✅ THIS is the connection

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "emotions": None, "input_text": ""}
    )

@app.post("/", response_class=HTMLResponse)
def analyze(request: Request, text: str = Form(...)):
    emotions = predict(text)   # 🔥 prediction.py runs HERE

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "emotions": emotions,
            "input_text": text
        }
    )
