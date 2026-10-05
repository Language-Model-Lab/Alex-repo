api.py contains the API endpoints for response generation, and is written using FastAPI.

schemas.py contains the main schemas for generations, interventions, and responses.

test_schemas.py contains test for correctness of all schemas.

model.py contains the model loading logic

interventions.py contains the functions that perform interventions on the model, like attention head ablation, MLP unit ablation, etc.