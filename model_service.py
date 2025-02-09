# model_service.py

import importlib
from typing import List, Dict, Any, Optional, Union
from dramatic_logger import DramaticLogger
import torch
import os
import pkgutil
from pydantic import BaseModel
import numpy as np

## =====================================-------------------=====================================
## ------------------------------------- CLASS DEFINITIONS -------------------------------------
## =====================================-------------------=====================================

class ModelParams:
    def __init__(self, request):
        # Make these instance variables to ensure __dict__ exists
        self.__dict__.update({
            'model': request.model,
            'input_type': request.input_type if hasattr(request, 'input_type') else "document",
            'pooling_strategy': request.pooling_strategy if hasattr(request, 'pooling_strategy') else "mean",
            'max_length': request.max_length if hasattr(request, 'max_length') else None,
            'task': request.task if hasattr(request, 'task') else None,
            'truncate_dim': request.truncate_dim if hasattr(request, 'truncate_dim') else None,
            'encoding_format': request.encoding_format if hasattr(request, 'encoding_format') else "float",
            'truncate': request.truncate if hasattr(request, 'truncate') else None
        })

    def __eq__(self, other):
        """Enable direct comparison of parameter objects"""
        if not isinstance(other, ModelParams):
            return False
        
        # Core parameters that always matter
        core_match = (
            self.model == other.model and
            self.pooling_strategy == other.pooling_strategy and
            self.max_length == other.max_length and
            self.task == other.task and
            self.truncate_dim == self.truncate_dim and
            self.encoding_format == other.encoding_format and
            self.truncate == other.truncate
        )
        
        # Only check input_type if it's not a model that handles input types via prefixing
        if not self.model.startswith(('mixedbread_ai', 'cl_nagoya', 'jinaai')):  # Add other prefix-based models as needed
            core_match = core_match and (self.input_type == other.input_type)
            
        return core_match

    def __str__(self):
        """String representation of parameters"""
        return (f"ModelParams(model={self.model}, input_type={self.input_type}, "
                f"pooling_strategy={self.pooling_strategy}, max_length={self.max_length}, "
                f"task={self.task}, truncate_dim={self.truncate_dim}, "
                f"encoding_format={self.encoding_format}, truncate={self.truncate})")

class EmbeddingOutput(BaseModel):
    embeddings: List[List[float]]
    dimensions: int

class ModelService:
    def __init__(self):
        self.current_handler = None
        self.model_initialized = False
        self.current_params = None
        self.metrics = {}  # Store metrics per model

    def _params_match(self, new_params: ModelParams) -> bool:
        """Check if current model parameters match the requested parameters."""
        matches = (
            self.model_initialized and
            self.current_params == new_params
        )
        if not matches:
            DramaticLogger["Dramatic"]["debug"](
                f"[ModelService] Parameters differ:\n"
                f"Current: {str(self.current_params)}\n"
                f"New: {str(new_params)}"
            )
        return matches

    def initialize_model(self, request):
        """
        Dynamically load a model handler based on request.model.
        """
        params = ModelParams(request)
        
        # Skip initialization if parameters match
        if self._params_match(params):
            DramaticLogger["Normal"]["info"](
                f"[ModelService] Model '{params.model}' already initialized with identical parameters. Skipping reinitialization."
            )
            return

        # Store new params
        self.current_params = params

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
                DramaticLogger["Dramatic"]["debug"](f"[ModelService] Using SentenceTransformer to encode ", texts)
                embeddings = self.current_handler.generate_embeddings(
                    texts,  # Pass texts directly
                    pooling_strategy=self.current_params.pooling_strategy,
                    task=self.current_params.task,
                    truncate_dim=self.current_params.truncate_dim
                )
            else:
                # For standard transformer models, use prepare_input
                inputs = self.current_handler.prepare_input(texts, self.current_params.input_type)
                # DramaticLogger["Dramatic"]["debug"](f"[ModelService] Using prepare_input to encode ", inputs)
                embeddings = self.current_handler.generate_embeddings(
                    inputs,
                    pooling_strategy=self.current_params.pooling_strategy,
                    task=self.current_params.task,
                    truncate_dim=self.current_params.truncate_dim
                )

            # Get dimensions from the first embedding
            dimensions = len(embeddings[0])

            # Update metrics
            self._update_metrics(len(texts), dimensions)

            return EmbeddingOutput(
                embeddings=embeddings,
                dimensions=dimensions
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
            "model_name": self.current_params.model if self.model_initialized else "none",
            "current_config": {
                "input_type": self.current_params.input_type,
                "pooling_strategy": self.current_params.pooling_strategy,
                "max_length": self.current_params.max_length,
                "task": self.current_params.task,
                "truncate_dim": self.current_params.truncate_dim
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
        if self.current_handler and self.current_params.model == model_name:
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
        if self.current_params.model not in self.metrics:
            self.metrics[self.current_params.model] = {
                "total_requests": 0,
                "total_texts": 0,
                "total_dimensions": embedding_dim,
            }
        
        self.metrics[self.current_params.model]["total_requests"] += 1
        self.metrics[self.current_params.model]["total_texts"] += num_texts
        self.metrics[self.current_params.model]["total_dimensions"] += embedding_dim 