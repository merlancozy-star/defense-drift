"""Embedder wrapper — supports Qwen3-Embedding-4B and generic sentence-transformers.

Encoding is done via mean pooling of last hidden states (standard for decoder-only
embedders) or via sentence-transformers API.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmbedderWrapper:
    """Text embedder wrapping HF models.

    Supports two modes:
    1. HF AutoModel with mean pooling (Qwen3-Embedding style)
    2. sentence-transformers API (for BGE, E5, etc.)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        batch_size: int = 32,
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._use_sentence_transformers = False
        self._embedding_dim = None

    def load(self):
        """Lazy-load the embedder model."""
        if self._loaded:
            return

        logger.info(f"Loading embedder from {self.model_path}...")

        # Try sentence-transformers first (cleaner API)
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(
                self.model_path,
                device=self.device,
                trust_remote_code=True,
            )
            self._use_sentence_transformers = True
            try:
                self._embedding_dim = self._st_model.get_embedding_dimension()
            except AttributeError:
                self._embedding_dim = self._st_model.get_sentence_embedding_dimension()
            self._loaded = True
            logger.info(
                f"Embedder loaded via sentence-transformers (dim={self._embedding_dim})"
            )
            return
        except Exception as e:
            logger.info(f"sentence-transformers not available ({e}), falling back to HF AutoModel")

        # Fall back to HF AutoModel with mean pooling
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            self.model_path,
            dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Determine embedding dimension
        self._embedding_dim = self.model.config.hidden_size
        self._loaded = True
        logger.info(f"Embedder loaded via HF AutoModel on {self.device} (dim={self._embedding_dim})")

    def unload(self):
        """Free GPU memory."""
        if self._use_sentence_transformers:
            del self._st_model
            self._st_model = None
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        torch.cuda.empty_cache()

    @property
    def embedding_dim(self) -> Optional[int]:
        return self._embedding_dim

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts to embeddings.

        Args:
            texts: List of text strings.
            normalize: L2-normalize embeddings (recommended for cosine similarity).
            show_progress: Show progress bar.

        Returns:
            numpy array of shape [len(texts), embedding_dim].
        """
        if not self._loaded:
            self.load()

        if self._use_sentence_transformers:
            embeddings = self._st_model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            return embeddings if isinstance(embeddings, np.ndarray) else np.array(embeddings)

        # HF AutoModel path with mean pooling
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            outputs = self.model(**enc)
            # Mean pooling over token dimension (excluding padding)
            hidden = outputs.last_hidden_state  # [B, T, D]
            mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, D]

            if normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)

            all_embeddings.append(pooled.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query."""
        emb = self.encode([query], normalize=True)
        return emb[0]

    def encode_passages(self, passages: List[str]) -> np.ndarray:
        """Encode a list of passages."""
        return self.encode(passages, normalize=True)

    def is_loaded(self) -> bool:
        return self._loaded
