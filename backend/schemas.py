from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field

class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length = 1, max_length=2000)
    configuration_a: ExperimentConfiguration
    configuration_b: ExperimentConfiguration


class GenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    condition_a_text: str
    condition_b_text: str
    condition_a_latency_ms: float
    condition_b_latency_ms: float

class ExperimentConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation: GenerationSettings
    intervention: InterventionSettings | None = None
    
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
