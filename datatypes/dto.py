from pydantic import BaseModel

class QuestionAnsweringParam(BaseModel):
    query: str
    user_id: str
    chat_id: str
    
class ChatHistoryParam(BaseModel):
    user_id: str
    chat_id: str
    
class IndexingIdParam(BaseModel):
    indexing_id: str