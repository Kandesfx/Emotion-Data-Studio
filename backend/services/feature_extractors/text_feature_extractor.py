"""Word-level PhoBERT text feature extractor — generates (T, 768) embeddings per clip.

Loads vinai/PhoBERT-base and encodes each word in the transcript.
Output shape matches MMSA DataLoader expectation: (MAX_SEQ_LEN, 768), float32.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

MAX_SEQ_LEN = 50
TEXT_DIM = 768


class TextFeatureExtractor:
    """Lazily imports torch/transformers so this module can be imported (e.g. by
    pipeline_orchestrator) without those heavy deps installed — only actual
    feature extraction needs them."""

    def __init__(self, output_dir: Path | None = None, device: str | None = None):
        self.output_dir = output_dir
        self._tokenizer: Any = None
        self._model: Any = None
        self._device_override = device
        self._device: str | None = None

    @property
    def device(self) -> str:
        if self._device is None:
            if self._device_override is not None:
                self._device = self._device_override
            else:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("vinai/PhoBERT-base")
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained("vinai/PhoBERT-base")
            self._model.to(self.device)
            self._model.eval()
        return self._model

    def extract_features(
        self,
        transcript: str,
        clip_id: str,
        word_timestamps: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Extract word-level PhoBERT embeddings for a transcript.

        Args:
            transcript: Raw transcript string from Whisper.
            clip_id: Clip identifier, used for output filename.
            word_timestamps: Optional list of {"word": str, "start": float, "end": float}.
                             Used to order embeddings; if None, split transcript by spaces.

        Returns:
            {
                "features": np.ndarray (MAX_SEQ_LEN, 768), float32,
                "feature_path": str,
                "shape": str,
                "num_words": int,
                "aligned": bool,
            }
        """
        if not transcript or not transcript.strip():
            return self._empty_result()

        # Build ordered list of (word, start, end)
        if word_timestamps:
            words = [(w["word"], float(w.get("start", 0)), float(w.get("end", 0))) for w in word_timestamps]
        else:
            words = [(w, 0.0, 0.0) for w in transcript.split()]

        # Filter empty tokens
        words = [(w, s, e) for w, s, e in words if w and len(w.strip()) > 0]
        if not words:
            return self._empty_result()

        word_texts = [w for w, _, _ in words]

        # Encode with PhoBERT (subword tokenization → aggregate subword vectors)
        embeddings = self._encode_words(word_texts)
        # embeddings: (num_words, 768) float32

        # Resample / pad to MAX_SEQ_LEN
        if embeddings.shape[0] > MAX_SEQ_LEN:
            # Truncate
            embeddings = embeddings[:MAX_SEQ_LEN]
        elif embeddings.shape[0] < MAX_SEQ_LEN:
            # Zero-pad
            pad_len = MAX_SEQ_LEN - embeddings.shape[0]
            pad = np.zeros((pad_len, TEXT_DIM), dtype=np.float32)
            embeddings = np.vstack([embeddings, pad])

        # Save
        output_dir = self.output_dir
        if output_dir is None:
            output_dir = Path.cwd() / "data" / "text_features"
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_path = output_dir / f"{clip_id}_text_features.npy"
        np.save(str(feature_path), embeddings.astype(np.float32))

        return {
            "features": embeddings,
            "feature_path": str(feature_path.resolve()),
            "shape": f"({embeddings.shape[0]}, {embeddings.shape[1]})",
            "num_words": embeddings.shape[0],
            "aligned": word_timestamps is not None,
        }

    def _encode_words(self, word_texts: list[str]) -> np.ndarray:
        """Encode each word via PhoBERT subword aggregation (mean pooling)."""
        import torch

        tokenizer = self.tokenizer
        model = self.model
        embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for word in word_texts:
                # Tokenize the single word
                inputs = tokenizer(
                    word,
                    return_tensors="pt",
                    truncation=True,
                    max_length=32,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = model(**inputs)
                # last_hidden_state: (1, num_subwords, 768)
                last_hidden = outputs.last_hidden_state[0]  # (num_subwords, 768)
                # Mean-pool subword tokens
                word_vec = last_hidden.mean(dim=0).cpu().numpy()   # (768,)
                embeddings.append(word_vec)

        return np.stack(embeddings, axis=0) if embeddings else np.zeros((0, TEXT_DIM), dtype=np.float32)

    def _empty_result(self) -> dict[str, Any]:
        zeros = np.zeros((MAX_SEQ_LEN, TEXT_DIM), dtype=np.float32)
        return {
            "features": zeros,
            "feature_path": "",
            "shape": f"({MAX_SEQ_LEN}, {TEXT_DIM})",
            "num_words": 0,
            "aligned": False,
            "warning": "empty_transcript",
        }
