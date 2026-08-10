from pydantic import BaseModel


class FinalPost(BaseModel):
    topic: str
    draft: str
    final_text: str
