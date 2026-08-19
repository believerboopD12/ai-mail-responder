from contextlib import asynccontextmanager
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

load_dotenv()

from app.classifier import IntentClassifier, detect_priority
from app.database import ProcessedEmail, SessionLocal, create_tables
from app.reply_service import ReplyService
from app.schemas import EmailListResponse, EmailRequest, EmailResponse, PredictionResponse

@asynccontextmanager
async def lifespan(_: FastAPI):
    create_tables()
    yield


app = FastAPI(title="AI Mail Responder", version="1.0.0", lifespan=lifespan)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@lru_cache
def get_classifier() -> IntentClassifier:
    return IntentClassifier()


def classify(email: EmailRequest) -> PredictionResponse:
    try:
        intent, confidence = get_classifier().predict(email.subject, email.body)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(intent=intent, confidence=confidence, priority=detect_priority(email.subject, email.body, intent))


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "name": "AI Mail Responder",
        "status": "running",
        "health": "/health",
        "docs": "/docs",
    }

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(email: EmailRequest) -> PredictionResponse:
    return classify(email)


@app.post("/reply", response_model=EmailResponse, status_code=status.HTTP_201_CREATED)
def reply(email: EmailRequest, db: Session = Depends(get_db)) -> ProcessedEmail:
    prediction = classify(email)
    try:
        draft = ReplyService().generate(email.subject, email.body, prediction.intent, prediction.priority)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Reply provider failed") from exc
    record = ProcessedEmail(subject=email.subject, body=email.body, predicted_intent=prediction.intent, confidence=prediction.confidence, priority=prediction.priority, generated_reply=draft)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@app.get("/emails", response_model=EmailListResponse)
def list_emails(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0), db: Session = Depends(get_db)) -> EmailListResponse:
    total = db.scalar(select(func.count()).select_from(ProcessedEmail)) or 0
    items = list(db.scalars(select(ProcessedEmail).order_by(ProcessedEmail.id.desc()).limit(limit).offset(offset)))
    return EmailListResponse(items=items, total=total)


@app.get("/emails/{email_id}", response_model=EmailResponse)
def get_email(email_id: int, db: Session = Depends(get_db)) -> ProcessedEmail:
    record = db.get(ProcessedEmail, email_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Email not found")
    return record
