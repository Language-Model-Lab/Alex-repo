from schemas import (
   GenerationSettings,
   InterventionSettings,
   GenerateRequest,
   GenerateResponse,
   ExperimentConfiguration
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

   generation_a= GenerationSettings(
      max_new_tokens=75, 
      temperature=0.25, 
      top_k=11, 
      top_p=0.25, 
      repetition_penalty=1.55
      )

   intervention_a=InterventionSettings(
      type="mlp", 
      layers=[0,1,2,3],
      proportion=0.25,
      seed=42,
      )

   generation_b= GenerationSettings(
      max_new_tokens=75, 
      temperature=0.5, 
      top_k=22, 
      top_p=0.5, 
      repetition_penalty=1.3
      )

   intervention_b=InterventionSettings(
      type="mlp", 
      layers=[0,1,2],
      proportion=0.45,
      seed=100,
      )

   config_a = ExperimentConfiguration(
               generation=generation_a, 
               intervention=intervention_a
               )
   config_b = ExperimentConfiguration(
               generation=generation_b, 
               intervention=intervention_b
               )

   request = GenerateRequest(
      prompt="The capital of France is",
      configuration_a= config_a,
      configuration_b=config_b,
   )

   assert request.prompt == "The capital of France is"
   assert request.configuration_a == config_a
   assert request.configuration_b == config_b

def test_valid_request_no_int():
   generation_a= GenerationSettings(
      max_new_tokens=75, 
      temperature=0.25, 
      top_k=11, 
      top_p=0.25, 
      repetition_penalty=1.55
      )

   generation_b= GenerationSettings(
      max_new_tokens=75, 
      temperature=0.5, 
      top_k=22, 
      top_p=0.5, 
      repetition_penalty=1.3
      )

   config_a = ExperimentConfiguration(generation=generation_a)
   config_b = ExperimentConfiguration(generation=generation_b)

   request = GenerateRequest(
      prompt="The capital of France is",
      configuration_a= config_a,
      configuration_b=config_b,
   )

   assert request.prompt == "The capital of France is"
   assert request.configuration_a.intervention is None
   assert request.configuration_b.intervention is None
   assert request.configuration_a == config_a
   assert request.configuration_b == config_b


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
      condition_a_text= "paodpasodsd",
      condition_b_text= "asdkasod2ed",
      condition_a_latency_ms= 2.32,
      condition_b_latency_ms= 3.32,
   )

   assert response.condition_a_text == "paodpasodsd"
   assert response.condition_b_text == "asdkasod2ed"
   assert response.condition_a_latency_ms == 2.32
   assert response.condition_b_latency_ms == 3.32

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