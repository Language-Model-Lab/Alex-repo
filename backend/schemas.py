from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field

class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = Field(default=50, ge=1, le=100)
    do_sample: bool = False
    temperature: float = Field(default=0.8, gt=0.0, le=10) 
    top_k: int = Field(default=50, ge=0, le=100)
    top_p: float = Field(default=0.95, ge=0.0, le=1)
    repetition_penalty: float = Field(default=1.1, gt=0.0, le=50)
    enable_thinking: bool = False

class InterventionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    type: Literal['mlp', 'attention']
    layers: list[Annotated[int, Field(ge=0)]] = Field(min_length=1)
    proportion: float = Field(default=0.25, ge=0.0, le=1.0)
    seed: int = 42

class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length = 1, max_length=2000)
    generation: GenerationSettings
    intervention: InterventionSettings | None = None

class GenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    baseline_text: str
    experimental_text: str
    baseline_latency_ms: float
    experimental_latency_ms: float
