import time
startup_start = time.perf_counter()

from fastapi import FastAPI
from schemas import GenerateRequest, GenerateResponse
from model import load_model
from generation import generate_text

app = FastAPI()

#Load model only once when backend is started
model, tokenizer = load_model("Qwen/Qwen3-0.6B")
print(f"Startup through model loading: {time.perf_counter() - startup_start:.2f}s")

@app.post("/generate", response_model=GenerateResponse)
def generate(request:GenerateRequest):

    settings = request.generation
    prompt = request.prompt
    generated_text = generate_text(model, tokenizer, prompt, settings)

    return GenerateResponse(
        baseline_text= generated_text,
        experimental_text= generated_text,
        baseline_latency_ms=3.33,
        experimental_latency_ms=123.123,
    )