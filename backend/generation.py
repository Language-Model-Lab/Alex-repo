from transformers import AutoTokenizer, AutoModelForCausalLM
from schemas import *

def run_generation(model, tokenizer, request:ExperimentConfiguration):
    shared_prompt = request.prompt
    

def generate_text(model, tokenizer, prompt:str, settings: GenerationSettings) -> str:
    messages = [
    {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=settings.enable_thinking,
    ).to(model.device)

    generation_kwargs = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": settings.do_sample,
        "repetition_penalty": settings.repetition_penalty,
        }

    if settings.do_sample:
        generation_kwargs["temperature"] = settings.temperature
        generation_kwargs["top_k"] = settings.top_k
        generation_kwargs["top_p"] = settings.top_p
    
    output = model.generate(
        **inputs,
        **generation_kwargs,
        )

    return tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
            )


     