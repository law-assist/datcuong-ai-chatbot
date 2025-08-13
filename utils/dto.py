from pydantic import BaseModel

class QueryQuestion(BaseModel):
    query: str