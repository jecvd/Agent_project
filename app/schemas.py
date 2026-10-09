from pydantic import BaseModel

class QARequest(BaseModel):
    question:str

class Reference(BaseModel):
    title:str
    source_url:str
    score:float

class QAResponse(BaseModel):
    answer:str
    references:list[Reference]

class AgentRequest(BaseModel):
    question: str
    session_id: str = "default"

