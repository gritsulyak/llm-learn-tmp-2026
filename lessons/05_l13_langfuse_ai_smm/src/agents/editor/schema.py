from typing import Literal

from pydantic import BaseModel


class EditorVerdict(BaseModel):
    verdict: Literal["ok", "revise"]
    comments: str = ""
