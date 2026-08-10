from pydantic import BaseModel, Field


class PlanItem(BaseModel):
    day: str
    topic: str
    format: str
    cta_hint: str


class ContentPlan(BaseModel):
    content_plan: list[PlanItem] = Field(default_factory=list)
