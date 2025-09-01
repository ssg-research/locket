import logging

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class NomicEmbeddingModel:
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        max_seq_length: int = 8192,
        logger: logging.Logger = None,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.logger = logger

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, device_map="auto"
        )

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text):
        try:
            single_input = False
            if isinstance(text, str):
                text = [text]
                single_input = True

            # Tokenize inputs
            inputs = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self.mean_pooling(outputs, inputs["attention_mask"])

                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)

            # Convert to list
            embeddings = embeddings.cpu().numpy().tolist()

            if single_input and len(embeddings) == 1:
                return embeddings[0]
            return embeddings

        except Exception as e:
            if self.logger:
                self.logger.error(f"Nomic embedding error: {e}", exc_info=True)
            else:
                print(f"Nomic embedding error: {e}")
            return None

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
