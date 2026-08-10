from pydantic import BaseModel


class DraftPost(BaseModel):
    topic: str
    draft_text: str
