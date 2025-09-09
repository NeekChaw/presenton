from typing import List, Optional
from pydantic import BaseModel, Field


class CreatePresentationRequest(BaseModel):
    content: str = Field(..., description="The content for the presentation")
    n_slides: int = Field(..., description="Number of slides to generate")
    language: str = Field(..., description="Language for the presentation")
    file_paths: Optional[List[str]] = Field(default=None, description="File paths for additional content")
    tone: Optional[str] = Field(default=None, description="The tone for the presentation")
    verbosity: Optional[str] = Field(default=None, description="The verbosity for the presentation")
    instructions: Optional[str] = Field(default=None, description="Additional instructions for the presentation")