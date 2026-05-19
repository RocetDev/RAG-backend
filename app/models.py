from pydantic import BaseModel
from typing import List, Dict


# think more about this models
class ChatResponse(BaseModel):
    message: str

class ChatRequest(BaseModel):
    user_id: str
    message: str


class SaveDocRequest(BaseModel):
    documents: List[Dict]
    user_id: str
    admin: bool

'''
{
  "documents": [
    {
      "text": "Пить газировку очень опасно! Она вызывает привыкание и сахарный диабет! Если, например, пить каждый день на протяжении 2 лет, то человек значительно повышает шансы на появления диабета!"
    }
  ],
  "user_id": 0,
  "admin": true
}
'''