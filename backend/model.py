from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name: str, device_map: str="auto"):
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
   return model, tokenizer