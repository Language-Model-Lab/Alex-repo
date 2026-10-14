from transformers import AutoTokenizer, AutoModelForCausalLM
from interventions import add_head_ablation, add_mlp_ablation
from schemas import (
   GenerationSettings,
   InterventionSettings,
   GenerateRequest,
   GenerateResponse,
   ExperimentConfiguration
   )


def run_generation(model, tokenizer, prompt: str, request:ExperimentConfiguration):
    generation_settings = request.generation
    intervention_settings = request.intervention
    handles = []

    try:
        if intervention_settings is not None:
            if intervention_settings.type == "mlp":
                for layer_idx in intervention_settings.layers:
                    handle, _, _, _ = add_mlp_ablation(
                        model=model, 
                        layer_idx=layer_idx, 
                        proportion= intervention_settings.proportion, 
                        seed=intervention_settings.seed
                        )
                    handles.append(handle)
            elif intervention_settings.type == "attention":
                for layer_idx in intervention_settings.layers:
                    handle, _, _ = add_head_ablation(
                        model=model, 
                        layer_idx=layer_idx, 
                        head_indices= intervention_settings.head_idxs, 
                        )
                    handles.append(handle)
                        
        model_response = generate_text(
                                model=model, 
                                tokenizer=tokenizer,
                                prompt=prompt, 
                                settings=generation_settings
                                )
    finally:
        for handle in handles:
            handle.remove()
            
    return model_response
    

    

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


     