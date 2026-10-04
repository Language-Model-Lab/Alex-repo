from schemas import (
   GenerationSettings,
   InterventionSettings,
   GenerateRequest,
   GenerateResponse
   )

import pytest
from pydantic import ValidationError

def test_valid_generation_settings():
   settings = GenerationSettings(
      max_new_tokens=75, 
      temperature=0.5, 
      top_k=22, 
      top_p=0.5, 
      repetition_penalty=1.3
      )
   
   assert settings.max_new_tokens == 75
   assert settings.do_sample == False
   assert settings.temperature == 0.5
   assert settings.top_k == 22
   assert settings.top_p == 0.5
   assert settings.repetition_penalty == 1.3
   assert settings.enable_thinking == False


def test_invalid_max_tokens():
      with pytest.raises(ValidationError):
        GenerationSettings(
      max_new_tokens=200
      )

def test_invalid_top_p():
      with pytest.raises(ValidationError):
        GenerationSettings(
      top_p=1.01
      )

def test_valid_intervention_settings():
   intervention = InterventionSettings(
      type="mlp", 
      layers=[0,1,2],
      proportion=0.45,
      seed=100,
      )
   
   assert intervention.type == "mlp"
   assert intervention.layers == [0,1,2]
   assert intervention.proportion == 0.45
   assert intervention.seed == 100

def test_invalid_intervention_type():
   with pytest.raises(ValidationError):
      InterventionSettings(
         type="mlb", 
         layers=[0,1,2],
      )   

def test_invalid_intervention_layers():
   with pytest.raises(ValidationError):
      InterventionSettings(
         type="mlp", 
         layers=[],
      )   

def test_valid_request():

   generation= GenerationSettings(
      max_new_tokens=75, 
      temperature=0.5, 
      top_k=22, 
      top_p=0.5, 
      repetition_penalty=1.3
      )

   intervention=InterventionSettings(
      type="mlp", 
      layers=[0,1,2],
      proportion=0.45,
      seed=100,
      )

   request = GenerateRequest(
      prompt="The capital of France is",
      generation= generation,
      intervention=intervention,
   )

   assert request.prompt == "The capital of France is"
   assert request.generation == generation
   assert request.intervention == intervention

def test_valid_request_no_int():
   request = GenerateRequest(
      prompt="The capital of France is",
      generation= GenerationSettings(
      max_new_tokens=75, 
      temperature=0.5, 
      top_k=22, 
      top_p=0.5, 
      repetition_penalty=1.3
      ),
   )

   assert request.prompt == "The capital of France is"
   assert request.intervention is None


def test_invalid_request_no_prompt():
   with pytest.raises(ValidationError):
      GenerateRequest(
         prompt="",
         generation= GenerationSettings(
            max_new_tokens=75, 
            temperature=0.5, 
            top_k=22, 
            top_p=0.5, 
            repetition_penalty=1.3
         ),
      )

def test_valid_response():
   response = GenerateResponse(
      baseline_text= "paodpasodsd",
      experimental_text= "asdkasod2ed",
      baseline_latency_ms= 2.32,
      experimental_latency_ms= 3.32,
   )

   assert response.baseline_text == "paodpasodsd"
   assert response.experimental_text == "asdkasod2ed"
   assert response.baseline_latency_ms == 2.32
   assert response.experimental_latency_ms == 3.32

def test_generation_settings_extra_fields():
   with pytest.raises(ValidationError):
      GenerationSettings(
      max_new_tokens=75, 
      temperature=0.5, 
      top_k=22, 
      top_p=0.5, 
      repetition_penalty=1.3,
      extra_field = 22,
      )

def test_intervention_settings_extra_fields():
   with pytest.raises(ValidationError):
      InterventionSettings(
         type="mlp", 
         layers=[0,1,2],
         proportion=0.45,
         seed=100,
         extra_field=23,
      )

def test_request_extra_fields():
   generation= GenerationSettings(
      max_new_tokens=75, 
      temperature=0.5, 
      top_k=22, 
      top_p=0.5, 
      repetition_penalty=1.3
      )

   intervention=InterventionSettings(
      type="mlp", 
      layers=[0,1,2],
      proportion=0.45,
      seed=100,
      )

   with pytest.raises(ValidationError):
      GenerateRequest(
         prompt="The capital of France is",
         generation= generation,
         intervention=intervention,
         extra_field=25,
   )

def test_response_extra_fields():
   with pytest.raises(ValidationError):
      GenerateResponse(
         baseline_text= "paodpasodsd",
         experimental_text= "asdkasod2ed",
         baseline_latency_ms= 2.32,
         experimental_latency_ms= 3.32,
         extra_field=23,
      )

test_valid_generation_settings()
test_invalid_max_tokens()
test_invalid_top_p()
test_valid_intervention_settings()
test_invalid_intervention_type()
test_invalid_intervention_layers()
test_valid_request()
test_valid_request_no_int()
test_invalid_request_no_prompt()
test_valid_response()
test_generation_settings_extra_fields()
test_intervention_settings_extra_fields()
test_request_extra_fields()
test_response_extra_fields()