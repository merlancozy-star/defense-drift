"""Generator wrapper for Qwen3-8B (or any HF causal LM).

Key responsibilities:
1. Token-level NLL computation (→ PerplexityScorer)
2. Generation with attention extraction (→ AttentionVarianceScorer)
3. Optional: synthetic poison passage generation (→ Injectors)

Must expose output_attentions=True to satisfy AttentionVarianceScorer's
white-box requirement.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class GeneratorWrapper:
    """Wraps a HuggingFace CausalLM for scoring and generation.

    Configuration is read from the ExperimentConfig's generator ModelConfig.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        max_length: int = 4096,
        output_attentions: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.output_attentions = output_attentions

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Lazy-load the model and tokenizer (expensive, done once)."""
        if self._loaded:
            return

        logger.info(f"Loading generator from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
            output_attentions=self.output_attentions,
        )
        self.model.eval()
        self._loaded = True
        logger.info(f"Generator loaded on {self.device}")

    def unload(self):
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        torch.cuda.empty_cache()

    @torch.no_grad()
    def compute_nll(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> List[float]:
        """Compute token-level negative log-likelihood (NLL) for each text.

        NLL = -1/|tokens| * Σ log P(token_i | context_{<i})

        This is used by PerplexityScorer: PPL = exp(NLL).
        Higher NLL → less likely under the model → more suspicious.

        Args:
            texts: List of passage texts.
            batch_size: Batch size for forward pass.

        Returns:
            List of per-text NLL values (scalars).
        """
        if not self._loaded:
            self.load()

        nlls = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            outputs = self.model(**enc)
            logits = outputs.logits  # [B, T, V]

            # Shift: predict token_{t+1} from position t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = enc["input_ids"][:, 1:].contiguous()

            # Per-token NLL
            token_nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)  # [B, T-1]

            # Mask padding
            mask = (shift_labels != self.tokenizer.pad_token_id).float()

            # Mean NLL over non-padding tokens
            for j in range(len(batch)):
                valid_tokens = mask[j].sum().item()
                if valid_tokens > 0:
                    mean_nll = (token_nll[j] * mask[j]).sum().item() / valid_tokens
                else:
                    mean_nll = float("inf")
                nlls.append(mean_nll)

        return nlls

    @torch.no_grad()
    def compute_perplexity(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """Compute perplexity for each text. PPL = exp(NLL)."""
        nlls = self.compute_nll(texts, batch_size)
        return [float(torch.exp(torch.tensor(n)).item()) for n in nlls]

    @torch.no_grad()
    def generate_with_attention(
        self,
        prompt: str,
        context_passages: List[str],
        max_new_tokens: int = 64,
    ) -> Dict:
        """Generate text and extract cross-passage attention weights.

        This is the CORE method for AttentionVarianceScorer.
        The model sees: [context_passages...] + prompt, and we extract
        attention weights from the generated tokens back to each passage.

        Args:
            prompt: The query/question.
            context_passages: Retrieved passages placed in context.
            max_new_tokens: Number of tokens to generate.

        Returns:
            dict with:
                - generated_text: str
                - attention_weights: List[torch.Tensor] — per-layer attention
                  of shape [n_layers, n_heads, gen_tokens, total_len]
                - passage_boundaries: List[(start, end)] token indices
                  for each passage in the full input
        """
        if not self._loaded:
            self.load()
        if not self.output_attentions:
            raise RuntimeError(
                "Generator loaded without output_attentions=True — "
                "AttentionVarianceScorer will fail. Re-initialize with output_attentions=True."
            )

        # Build input: passages + prompt
        passage_texts = "\n\n".join(
            f"[Passage {i+1}]: {p}" for i, p in enumerate(context_passages)
        )
        full_input = f"{passage_texts}\n\n[Question]: {prompt}\n[Answer]:"

        inputs = self.tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length - max_new_tokens,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        # Tokenize each passage separately to get boundaries
        passage_boundaries = []
        cursor = 0
        for p in context_passages:
            p_tokens = self.tokenizer(
                f"[Passage]: {p}",
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            p_len = p_tokens["input_ids"].shape[1]
            # Approximate boundaries (will be refined by actual tokenization)
            passage_boundaries.append((cursor, cursor + p_len))
            cursor += p_len

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_ids = outputs.sequences[0][input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract attention: tuple of tuples
        # Outer tuple = generation steps, inner tuple = layers
        # Each attention tensor: [B, n_heads, 1, full_seq_len]
        attention_weights = []
        if outputs.attentions is not None:
            for step_attn in outputs.attentions:
                # step_attn is tuple of [B, n_heads, 1, full_seq_len] per layer
                layer_attns = [a[0].cpu() for a in step_attn]  # list of [n_heads, 1, full_seq_len]
                # Stack to [n_layers, n_heads, 1, full_seq_len]
                attention_weights.append(torch.stack(layer_attns))

        return {
            "generated_text": generated_text,
            "attention_weights": attention_weights,  # list of [n_layers, n_heads, 1, full_seq_len]
            "passage_boundaries": passage_boundaries,
            "input_len": input_len,
        }

    def is_loaded(self) -> bool:
        return self._loaded
