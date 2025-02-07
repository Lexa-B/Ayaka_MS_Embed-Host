from embeddings._base_model_handler import BaseModelHandler
from dramatic_logger import DramaticLogger
import torch
from typing import List, Dict, Any, Union

MODEL_PATH = "./Models/mixedbread-ai/mxbai-embed-large-v1"

class ModelHandler(BaseModelHandler):
    """
    Handler for MixedBread's large embedding model.
    Uses standard transformer architecture with query/document prefixing.
    """

    def build_model_path(self) -> str:
        return MODEL_PATH

    def preprocess_texts(self, texts: List[str], input_type: str) -> List[str]:
        """Add appropriate prefixes based on input type."""
        try:
            if input_type == "query":
                return [f'Instruct: Represent this query for searching relevant passages\nQuery: {text}' for text in texts]
            elif input_type in ["document", "passage"]:
                return texts  # No special prefix for documents
            else:
                raise ValueError(f"Unsupported input type: {input_type}")
        except Exception as e:
            DramaticLogger["Dramatic"]["error"]("[MixedBread Handler] Error preprocessing texts:", str(e))
            raise

    def generate_embeddings(self, inputs: Dict[str, torch.Tensor], 
                          pooling_strategy: str = "mean",
                          task: str | None = None,
                          truncate_dim: int | None = None) -> torch.Tensor:
        """Generate embeddings using standard transformer with mean pooling."""
        try:
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Apply pooling strategy
                if pooling_strategy == "mean":
                    attention_mask = inputs['attention_mask'].unsqueeze(-1)
                    token_embeddings = outputs.last_hidden_state
                    sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
                    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                elif pooling_strategy == "cls":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")

                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Handle dimension truncation if specified
                if truncate_dim is not None:
                    embeddings = embeddings[:, :truncate_dim]

                return embeddings

        except Exception as e:
            DramaticLogger["Dramatic"]["error"]("[MixedBread Handler] Error generating embeddings:", str(e))
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Return MixedBread-specific model information."""
        return {
            "dimensions": self.model.config.hidden_size,
            "max_sequence_length": self.model.config.max_position_embeddings,
            "model_type": "transformer",
            "description": "General purpose embedding model with query/document asymmetric encoding",
            "supported_languages": ["en", "multi"],  # Supports English and other languages
            "input_types": {
                "query": "Instruct: Represent this query for searching relevant passages\nQuery: {text}",
                "document": "Direct text, no prefix",
                "passage": "Direct text, no prefix"
            }
        } 