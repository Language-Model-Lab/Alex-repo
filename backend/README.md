api.py contains the API endpoints for response generation, and is written using FastAPI.
To run the web server and test API locally, use 
```uv run uvicorn api:app```

schemas.py contains the main schemas for generations, interventions, and responses.

test_schemas.py contains test for correctness of all schemas.
To run tests use
```uv run pytest -v```

model.py contains the model loading logic

generation.py contains the functions that call the model, generate text, and intervention wrappers

interventions.py contains the functions that perform interventions on the model, like attention head ablation, MLP unit ablation, etc.