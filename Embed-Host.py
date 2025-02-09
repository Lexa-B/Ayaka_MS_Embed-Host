from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from dramatic_logger import DramaticLogger
from model_service import ModelService
import json

app = FastAPI(
    title="Ayaka Embeddings API",
    description="A local Embedding model host microservice that serves the Ayaka Embedding models. It roughly mimics the OpenAI API parameters and API calls.",
    version="0.0.1",
)

favicon_path = 'favicon.ico'

# Initialize the model service
model_service = ModelService()

## ========================================------------========================================
## ---------------------------------------- MIDDLEWARE ----------------------------------------
## ========================================------------========================================

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        DramaticLogger["Normal"]["info"](f"[Embedding-Host] Request: {request.method} {request.url}")
        if request.method == "POST":
            body = await request.body()
            try:
                body_str = body.decode('utf-8')
                DramaticLogger["Dramatic"]["debug"]("[Embedding-Host] Request Body:", json.loads(body_str))
            except UnicodeDecodeError:
                DramaticLogger["Dramatic"]["warning"]("[Embedding-Host] Could not decode request body.")
            
            # Reassign the body so downstream can read it
            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}
            request = Request(scope=request.scope, receive=receive)
        
        if request.method == "GET":  # Development debugging only
            headers = dict(request.headers)
            DramaticLogger["Dramatic"]["debug"]("[Embedding-Host] GET Request Headers:", headers)
        
        response = await call_next(request)
        DramaticLogger["Normal"]["info"](f"[Embedding-Host] Response: {response.status_code}")
        return response

app.add_middleware(LoggingMiddleware)

## ========================================-----------------========================================
## ---------------------------------------- PYDANTIC MODELS ---------------------------------------
## ========================================-----------------========================================

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    input_type: str = "document"  # document, query, or passage
    encoding_format: str = "float"
    truncate: Optional[str] = None
    pooling_strategy: str = "mean"
    max_length: Optional[int] = None
    task: Optional[str] = None
    truncate_dim: Optional[int] = None

class EmbeddingData(BaseModel):
    index: int
    embedding: List[float]
    object: str = "embedding"

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]

class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "user"
    permission: List = []

class ModelsResponse(BaseModel):
    data: List[Model]
    object: str = "list"

## ========================================--------------========================================
## ---------------------------------------- BASIC ROUTES ----------------------------------------
## ========================================--------------========================================

@app.get("/")
def home():
    return {"message": "Welcome! Please GET /docs for API information"}

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)

## ========================================--------------========================================
## ---------------------------------------- MODEL ROUTES ----------------------------------------
## ========================================--------------========================================

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings for the provided input text."""
    try:
        # Initialize model if needed
        model_service.initialize_model(request)

        # Generate embeddings
        output = model_service.generate_embeddings(request.input)
        
        # Format response to exactly match OpenAI API spec
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": i
                }
                for i, embedding in enumerate(output.embeddings)
            ],
            "model": request.model,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0
            }
        }
        
    except ValueError as ve:
        if "Model files not found" in str(ve):
            raise HTTPException(status_code=503, detail=str(ve))
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        DramaticLogger["Dramatic"]["error"](f"[Embedding-Host] Error: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/v1/models")
async def list_models():
    """List available embedding models."""
    try:
        models = model_service.get_available_models()
        return ModelsResponse(data=models)
        
    except Exception as e:
        DramaticLogger["Dramatic"]["error"](f"[Embedding-Host] Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models.")

@app.get("/v1/models/{model_name}/info")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model."""
    try:
        return model_service.get_model_info(model_name)
        
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model info: {str(e)}")

@app.get("/v1/metrics")
async def get_metrics():
    """Get performance metrics for all models."""
    try:
        return model_service.get_metrics()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}") 