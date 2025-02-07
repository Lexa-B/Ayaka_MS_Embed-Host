from embeddings._base_model_handler import BaseModelHandler
from dramatic_logger import DramaticLogger
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any, Union

MODEL_PATH = "./Models/cl-nagoya/ruri-large"

class ModelHandler(BaseModelHandler):
    """
    Handler for Ruri Japanese embedding model.
    Requires sentence-transformers and Japanese language support.
    """

    def build_model_path(self) -> str:
        return MODEL_PATH

    def load_model(self):
        """Override to use SentenceTransformer instead of standard loading."""
        try:
            self.model = SentenceTransformer(
                self.model_path,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.tokenizer = None  # Not needed for SentenceTransformer
            DramaticLogger["Normal"]["info"]("[Ruri Handler] Model loaded successfully.")
        except Exception as e:
            if "No such file or directory" in str(e):
                DramaticLogger["Dramatic"]["warning"]("[Ruri Handler] Model files not found:", str(e))
                raise ValueError(f"Model files not found for {self.model_path}")
            else:
                DramaticLogger["Dramatic"]["error"]("[Ruri Handler] Error loading model:", str(e))
                raise

    def preprocess_texts(self, texts: List[str], input_type: str) -> List[str]:
        """Add appropriate Japanese prefixes based on input type."""
        try:
            if input_type == "query":
                return [f"クエリ: {text}" for text in texts]
            elif input_type in ["document", "passage"]:
                return [f"文章: {text}" for text in texts]
            else:
                raise ValueError(f"Unsupported input type for Ruri: {input_type}")
        except Exception as e:
            DramaticLogger["Dramatic"]["error"]("[Ruri Handler] Error preprocessing texts:", str(e))
            raise

    def prepare_input(self, texts: List[str], input_type: str) -> List[str]:
        """Override to skip tokenization as SentenceTransformer handles it internally."""
        return self.preprocess_texts(texts, input_type)

    def generate_embeddings(self, inputs: Union[List[str], Dict[str, torch.Tensor]], 
                          pooling_strategy: str = "mean",
                          task: str | None = None,
                          truncate_dim: int | None = None) -> torch.Tensor:
        """Generate embeddings using SentenceTransformer's encode method."""
        try:
            # For SentenceTransformer, we expect inputs to be a list of strings
            if not isinstance(inputs, list):
                raise ValueError("SentenceTransformer expects a list of strings as input")
            
            embeddings = self.model.encode(
                inputs,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

            # Handle dimension truncation if specified
            if truncate_dim is not None:
                embeddings = embeddings[:, :truncate_dim]

            return embeddings

        except Exception as e:
            DramaticLogger["Dramatic"]["error"]("[Ruri Handler] Error generating embeddings:", str(e))
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Return Ruri-specific model information."""
        return {
            "dimensions": 1024,
            "max_sequence_length": 512,
            "model_type": "sentence-transformer",
            "description": "Japanese general text embedding model",
            "supported_languages": ["ja"],
            "input_types": {
                "query": "クエリ: prefix",
                "document": "文章: prefix",
                "passage": "文章: prefix"
            }
        } 