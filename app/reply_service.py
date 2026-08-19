import os
from openai import OpenAI


class ReplyService:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, subject: str, body: str, intent: str, priority: str) -> str:
        if not self.api_key:
            return self._fallback(intent, priority)
        response = OpenAI(api_key=self.api_key).responses.create(
            model=self.model,
            instructions="Write only a concise, professional email reply draft. Do not invent facts, promises, names, or resolution dates. Ask for needed details.",
            input=f"Subject: {subject}\nBody: {body}\nIntent: {intent}\nPriority: {priority}",
        )
        return response.output_text.strip()

    @staticmethod
    def _fallback(intent: str, priority: str) -> str:
        openings = {
            "complaint": "Thank you for bringing this issue to our attention. We are sorry for the inconvenience.",
            "refund": "Thank you for contacting us about your refund. We understand your concern.",
            "support": "Thank you for contacting support. We are ready to help with this issue.",
            "sales": "Thank you for your interest in our services.",
            "inquiry": "Thank you for your question.",
            "spam": "Thank you for your message.",
        }
        urgency = " We will review it as a priority." if priority == "high" else ""
        return f"Hello,\n\n{openings.get(intent, openings['inquiry'])}{urgency} Please share any relevant order or account reference so the team can review it.\n\nBest regards,\nCustomer Support"
