# model_service.py

import importlib
from typing import List, Dict, Any, Optional, Union
from dramatic_logger import DramaticLogger
import torch
import os
import pkgutil
from pydantic import BaseModel

## =====================================-------------------=====================================
## ------------------------------------- CLASS DEFINITIONS -------------------------------------
## =====================================-------------------=====================================

class ModelParams:
    def __init__(self, request):
        self.model = request.model                            # Name of the model being used
        self.input_type = request.input_type                  # Type of input (document, query, passage)
        self.pooling_strategy = request.pooling_strategy      # Pooling strategy for embeddings
        self.max_length = request.max_length                  # Maximum sequence length
        self.task = request.task                             # Optional task for models that support it
        self.truncate_dim = request.truncate_dim             # Optional dimension truncation
        self.encoding_format = request.encoding_format        # Output format (float)

class EmbeddingOutput(BaseModel):
    embeddings: List[List[float]]
    dimensions: int

class ModelService:
    def __init__(self):
        self.current_handler = None
        self.model_initialized = False
        self.model_name = None
        self.input_type = None
        self.pooling_strategy = None
        self.max_length = None
        self.task = None
        self.truncate_dim = None
        self.metrics = {}  # Store metrics per model

    def initialize_model(self, request):
        """
        Dynamically load a model handler based on request.model.
        """
        params = ModelParams(request)
        self.model_name = params.model
        self.input_type = params.input_type
        self.pooling_strategy = params.pooling_strategy
        self.max_length = params.max_length
        self.task = params.task
        self.truncate_dim = params.truncate_dim

        # Transform model name for handler lookup (e.g., "jinaai_jina-embeddings-v3" => "jinaai_jina_embeddings_v3")
        sanitized_name = params.model.lower().replace('-', '_').replace('.', '_')
        handler_module_name = f"embeddings.model_srv_{sanitized_name}"

        try:
            handler_module = importlib.import_module(handler_module_name)
            handler_class = getattr(handler_module, "ModelHandler")
        except (ImportError, AttributeError) as e:
            DramaticLogger["Dramatic"]["error"](
                f"[ModelService] Could not find a handler for model '{params.model}':",
                f"Please ensure it is supported by the Embedding-Host. Error: {e}"
            )
            raise ValueError(f"Model not available: {params.model}")

        try:
            self.current_handler = handler_class(params)
            self.model_initialized = True
            DramaticLogger["Normal"]["success"](f"[ModelService] Model '{params.model}' initialized successfully.")
        except Exception as e:
            if "Model files not found" in str(e):
                DramaticLogger["Dramatic"]["warning"](f"[ModelService] Model files not found:", str(e))
                raise ValueError(f"Model files not found for {params.model}")
            else:
                DramaticLogger["Dramatic"]["error"](f"[ModelService] Failed to initialize model:", str(e))
            raise ValueError(f"Failed to initialize model: {params.model}")

    def generate_embeddings(self, texts: Union[str, List[str]]) -> EmbeddingOutput:
        """
        Generate embeddings from the loaded model using the current handler.
        """
        if not self.model_initialized or not self.current_handler:
            DramaticLogger["Dramatic"]["error"](f"[ModelService] Model not initialized.")
            raise ValueError("Model not initialized.")

        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]

        try:
            # For SentenceTransformer models, pass texts directly
            if hasattr(self.current_handler.model, 'encode'):
                embeddings = self.current_handler.generate_embeddings(
                    texts,  # Pass texts directly
                    pooling_strategy=self.pooling_strategy,
                    task=self.task,
                    truncate_dim=self.truncate_dim
                )
            else:
                # For standard transformer models, use prepare_input
                inputs = self.current_handler.prepare_input(texts, self.input_type)
                embeddings = self.current_handler.generate_embeddings(
                    inputs,
                    pooling_strategy=self.pooling_strategy,
                    task=self.task,
                    truncate_dim=self.truncate_dim
                )

            # Update metrics
            self._update_metrics(len(texts), embeddings.shape[1])

            return EmbeddingOutput(
                embeddings=embeddings.tolist(),
                dimensions=embeddings.shape[1]
            )

        except Exception as e:
            DramaticLogger["Dramatic"]["error"](f"[ModelService] Error generating embeddings:", str(e))
            raise

    def get_status(self) -> Dict[str, Any]:
        """
        Returns status, including whether a model is initialized and which model is loaded.
        """
        return {
            "status": "running",
            "model_initialized": self.model_initialized,
            "model_name": self.model_name if self.model_initialized else "none",
            "current_config": {
                "input_type": self.input_type,
                "pooling_strategy": self.pooling_strategy,
                "max_length": self.max_length,
                "task": self.task,
                "truncate_dim": self.truncate_dim
            } if self.model_initialized else None
        }

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Retrieves a list of available models with their metadata.
        """
        models = []
        embeddings_directory = os.path.join(os.path.dirname(__file__), 'embeddings')

        for finder, name, ispkg in pkgutil.iter_modules([embeddings_directory]):
            if name.startswith("model_srv_") and not ispkg:
                model_id = name.replace("model_srv_", "").replace("_", "-")
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": int(os.path.getctime(os.path.join(embeddings_directory, name + ".py"))),
                    "owned_by": "user",
                    "permission": []
                })

        return models

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        """
        if self.current_handler and self.model_name == model_name:
            return self.current_handler.get_model_info()
        
        # Try to load model info without fully initializing
        sanitized_name = model_name.lower().replace('-', '_').replace('.', '_')
        handler_module_name = f"embeddings.model_srv_{sanitized_name}"
        
        try:
            handler_module = importlib.import_module(handler_module_name)
            return handler_module.get_model_info()
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Model info not available for: {model_name}")

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for all models.
        """
        return self.metrics

    def _update_metrics(self, num_texts: int, embedding_dim: int):
        """
        Update metrics for the current model.
        """
        if self.model_name not in self.metrics:
            self.metrics[self.model_name] = {
                "total_requests": 0,
                "total_texts": 0,
                "total_dimensions": embedding_dim,
            }
        
        self.metrics[self.model_name]["total_requests"] += 1
        self.metrics[self.model_name]["total_texts"] += num_texts
        self.metrics[self.model_name]["total_dimensions"] += embedding_dim 