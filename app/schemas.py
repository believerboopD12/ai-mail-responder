from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmailRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=250)
    body: str = Field(min_length=1, max_length=10_000)

    @field_validator("subject", "body")
    @classmethod
    def reject_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class PredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_config = ConfigDict(populate_by_name=True)

    intent: str = Field(validation_alias="predicted_intent")
    confidence: float = Field(ge=0, le=1)
    priority: str


class EmailResponse(PredictionResponse):
    model_config = ConfigDict(from_attributes=True)
    id: int
    subject: str
    body: str
    generated_reply: str
    created_at: datetime


class EmailListResponse(BaseModel):
    items: list[EmailResponse]
    total: int
