from pydantic import BaseModel, Field
from typing import Optional

class VerifyResponse(BaseModel):
    score: float
    decision: bool


class TrainMicResponse(BaseModel):
    status: str
    message: str
