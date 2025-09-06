from pydantic import BaseModel

class QueryQuestion(BaseModel):
    query: str
    
class IndexMongoId(BaseModel):
    indexing_id: str