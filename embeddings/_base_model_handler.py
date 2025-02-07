# embeddings/_base_model_handler.py

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any, Union
import os
from dramatic_logger import DramaticLogger

# Remove dotenv dependency and just get token directly
HF_API_Token = os.getenv('HF_API_TOKEN')  # Note: Changed to match common env var naming

class BaseModelHandler:
    """Base class for embedding model handlers with shared logic."""

    def __init__(self, params):
        DramaticLogger["Normal"]["info"](f"[BaseModelHandler] Initialization called with model:", params.model)
        self.params = params
        self.model = None
        self.tokenizer = None

        try:
            self.model_path = self.build_model_path()
            self.load_model()
            DramaticLogger["Normal"]["info"](f"[BaseModelHandler] init done. Model path:", self.model_path)

        except Exception as e:
            if "Model files not found" in str(e):
                DramaticLogger["Dramatic"]["warning"]("[BaseModelHandler] Model files not found:", str(e))
                raise ValueError(f"Model files not found for {self.params.model}")
            else:
                DramaticLogger["Dramatic"]["error"](f"[BaseModelHandler] Error in initialization:", str(e))
                raise

    def build_model_path(self) -> str:
        """Build the path to the model files."""
        try:
            DramaticLogger["Dramatic"]["warning"](
                "[BaseModelHandler] No explicit build_model_path() override found; using default."
            )
            return f"./Models/{self.params.model}"
        except Exception as e:
            DramaticLogger["Dramatic"]["error"](
                "[BaseModelHandler] build_model_path() encountered an error:",
                str(e),
                exc_info=True
            )
            raise e

    def load_model(self):
        """Load the model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True
            )

            if torch.cuda.is_available():
                DramaticLogger["Normal"]["debug"]("[BaseModelHandler] GPU is available. Moving model to GPU.")
                self.model.to("cuda")
            DramaticLogger["Normal"]["info"]("[BaseModelHandler] Default load_model() done.")
        except Exception as e:
            if "Incorrect path_or_model_id: './Models/" in str(e):
                HubPath = str(e).split("'")[1].lstrip("./Models/")
                DramaticLogger["Dramatic"]["warning"]("[BaseModelHandler] Model files not found:", str(e))
                print(f"Model path: {self.model_path}")
                self.download_model(HubPath)
                raise ValueError(f"Model files not found for {HubPath}")
            else:
                DramaticLogger["Dramatic"]["error"]("[BaseModelHandler] Error loading model:", f"Error: {str(e)}")
                raise Exception(f"Failed to load model: {str(e)}")

    def download_model(self, model_hub_path: str):
        """Download model files from Hugging Face Hub."""
        if not HF_API_Token:
            DramaticLogger["Dramatic"]["error"]("[BaseModelHandler] Error getting API token:", "No HuggingFace API token found.")
            raise ValueError("No HuggingFace API token found. Set the HF_API_Token environment variable.")

        try:
            DramaticLogger["Normal"]["info"]("[BaseModelHandler] Starting model download from Hugging Face Hub:", model_hub_path)
            import threading
            download_thread = threading.Thread(
                target=self._download_model_thread,
                args=(model_hub_path,)
            )
            download_thread.daemon = True
            download_thread.start()
            raise ValueError(f"Model files not found for {model_hub_path}")
            
        except ValueError as ve:
            raise ve
        except Exception as e:
            DramaticLogger["Dramatic"]["error"]("[BaseModelHandler] Error downloading model files:", f"Error: {str(e)}")
            raise Exception(f"Failed to download model files: {str(e)}")

    def _download_model_thread(self, model_hub_path: str):
        """Helper method to handle the actual model download in a separate thread."""
        try:
            model = AutoModel.from_pretrained(
                model_hub_path,
                local_files_only=False,
                token=HF_API_Token
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_hub_path,
                local_files_only=False,
                token=HF_API_Token
            )
            model.save_pretrained(f"./Models/{model_hub_path}")
            tokenizer.save_pretrained(f"./Models/{model_hub_path}")
            DramaticLogger["Normal"]["info"]("[BaseModelHandler] Model files downloaded successfully.")
        except Exception as e:
            DramaticLogger["Dramatic"]["error"]("[BaseModelHandler] Error in download thread:", f"Error: {str(e)}")

    def prepare_input(self, texts: List[str], input_type: str) -> Dict[str, torch.Tensor]:
        """Prepare input texts for embedding generation."""
        try:
            processed_texts = self.preprocess_texts(texts, input_type)
            inputs = self.tokenizer(
                processed_texts,
                padding=True,
                truncation=True,
                max_length=self.params.max_length if self.params.max_length else 512,
                return_tensors="pt"
            )
            return inputs
        except Exception as e:
            DramaticLogger["Dramatic"]["error"]("[BaseModelHandler] Error preparing input:", str(e))
            raise

    def preprocess_texts(self, texts: List[str], input_type: str) -> List[str]:
        """Preprocess texts based on input type. Override in specific handlers."""
        return texts

    def generate_embeddings(self, inputs: Dict[str, torch.Tensor], 
                          pooling_strategy: str = "mean",
                          task: str | None = None,
                          truncate_dim: int | None = None) -> torch.Tensor:
        """Generate embeddings from input tensors."""
        try:
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
                embeddings = F.normalize(embeddings, p=2, dim=1)

                # Handle dimension truncation if specified
                if truncate_dim is not None:
                    embeddings = embeddings[:, :truncate_dim]

                return embeddings

        except Exception as e:
            DramaticLogger["Dramatic"]["error"]("[BaseModelHandler] Error generating embeddings:", str(e))
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information. Override in specific handlers."""
        return {
            "dimensions": self.model.config.hidden_size,
            "max_sequence_length": getattr(self.model.config, "max_position_embeddings", 512),
            "model_type": self.model.config.model_type,
            "description": "Base embedding model"
        } 