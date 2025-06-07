# /home/piero/Piero/AI/AI-Engineer/src/data_models.py
from pydantic import BaseModel, ConfigDict

class FileToCreate(BaseModel):
    path: str
    content: str
    model_config = ConfigDict(extra='ignore', frozen=True)

class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str
    model_config = ConfigDict(extra='ignore', frozen=True)