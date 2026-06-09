"""Vietnamese text emotion classifier — PhoBERT zero-shot + lexicon ensemble.

This module provides a text emotion model that:
1. Encodes the Vietnamese transcript using PhoBERT-base
2. Computes semantic similarity against 6 emotion prototype sentences
3. Combines with Vietnamese lexicon keyword matching for robustness
4. Returns 6 emotion scores normalized to [0, 1]

Emotion prototypes (hand-crafted Vietnamese sentences representing each emotion):
  happy:     "Tôi rất vui và hạnh phúc"      → high pitch words: vui, hạnh phúc, cười, sung sướng
  sad:       "Tôi buồn và đau lòng"           → low pitch words: buồn, khóc, cô đơn, nhớ
  angry:     "Tôi rất giận và tức giận"        → tense words: giận, tức, bực, ghét
  fear:      "Tôi rất sợ và lo lắng"           → fearful words: sợ, hoảng, nguy hiểm
  surprise:  "Thật bất ngờ, không thể tin được" → exclamatory words: sao, ơi, bất ngờ
  disgust:   "Thật kinh tởm và ghê"            → negative words: ghê, kinh, tởm, bẩn
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

EMOTION_CLASSES = ["happy", "sad", "angry", "fear", "surprise", "disgust"]

# Prototype sentences for each emotion (Vietnamese)
_EMOTION_PROTOTYPES = {
    "happy": [
        "Tôi rất vui và hạnh phúc",
        "Thật là tuyệt vời",
        "Tôi cảm thấy vui vẻ và hân hoan",
        "May quá, tôi rất hào hứng",
        "Cười đi, cuộc sống đẹp lắm",
    ],
    "sad": [
        "Tôi cảm thấy buồn và đau lòng",
        "Thật cô đơn và tủi hổ",
        "Nước mắt chảy dài",
        "Tôi nhớ và thương rất nhiều",
        "Thất vọng quá, không còn gì nữa",
    ],
    "angry": [
        "Tôi rất giận và tức ghê lên",
        "Đồ khốn nạn, im đi",
        "Ghét, căm hận vô cùng",
        "Điên lên được, không chịu được",
        "Tức quá, không thể tha thứ",
    ],
    "fear": [
        "Tôi rất sợ hãi và hoảng loạn",
        "Nguy hiểm quá, chạy đi thôi",
        "Tim đập rộn ràng vì lo lắng",
        "Đừng làm thế, tôi sợ lắm",
        "Hãi hùng và run sợ",
    ],
    "surprise": [
        "Thật bất ngờ, không thể tin được",
        "Sao có thể như vậy chứ",
        "Trời ơi, ơi là ơi",
        "Không thể nào, thật á",
        "Ủa, cái gì vậy",
    ],
    "disgust": [
        "Thật kinh tởm và ghê tởm",
        "Bẩn thỉu quá, khinh bỉ",
        "Tởm lắm, đáng ghét",
        "Ghê quá, không chịu được",
        "Kinh khủng và buồn nôn",
    ],
}

# Vietnamese emotion lexicon (fallback when PhoBERT encoding fails)
_VI_LEXICON = {
    "happy": ["vui", "hạnh phúc", "cười", "thích", "yêu", "tuyệt", "may quá", "mừng", "sướng", "hân hoan", "vui vẻ", "sung sướng", "hoan hỉ", "rộn ràng"],
    "sad": ["buồn", "khóc", "đau lòng", "cô đơn", "mất", "nhớ", "tủi", "thất vọng", "tủi nhục", "nước mắt", "bi lụy", "sầu", "thương"],
    "angry": ["giận", "tức", "bực", "đồ khốn", "im đi", "câm", "ghét", "điên", "không tha", "căm hận", " căm", "tức giận", "bực bội"],
    "fear": ["sợ", "lo", "hoảng", "cứu", "nguy hiểm", "chạy đi", "đừng", "hãi", "run", "rùng mình", "lo lắng", "bất an", "hoảng loạn"],
    "surprise": ["sao", "gi co", "that an", "khong the", "bat ngo", "troi oi", "ua", "chu sao", "lam sao", "that khong", "khong tin noi"],
    "disgust": ["ghe", "kinh", "tom", "ban", "khinh", "dang ghet", "buon non", "ghe tom", "kinh tom", "buc", "ngay", "ac"],
}


class VietnameseTextEmotionClassifier:
    """Vietnamese text emotion classifier using PhoBERT zero-shot + lexicon.

    The classifier uses PhoBERT embeddings to compute cosine similarity between
    the input transcript and emotion prototype sentences. It also incorporates
    Vietnamese keyword matching as a secondary signal.

    Falls back to lexicon-only scoring when PhoBERT is unavailable.
    """

    def __init__(self, device: str | None = None):
        if device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"
        else:
            self._device = device

        self._tokenizer = None
        self._model = None
        self._prototype_embeddings: dict[str, np.ndarray] = {}

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
            self._model.to(self._device)
            self._model.eval()
            self._encode_prototypes()
        return self._model

    def _encode_prototypes(self):
        """Pre-encode all emotion prototypes once."""
        try:
            import torch
        except Exception:
            return

        with torch.no_grad():
            for emotion, sentences in _EMOTION_PROTOTYPES.items():
                embeddings = []
                for sent in sentences:
                    inputs = self.tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    outputs = self._model(**inputs)
                    vec = outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()
                    embeddings.append(vec)
                self._prototype_embeddings[emotion] = np.mean(embeddings, axis=0)

    def _encode_text(self, text: str) -> np.ndarray | None:
        """Encode a transcript into a 768-dim PhoBERT embedding."""
        try:
            import torch
        except Exception:
            return None

        try:
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                return outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()
        except Exception:
            return None

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _lexicon_scores(self, text: str) -> dict[str, float]:
        """Score emotions by keyword matching with phrase-length weighting."""
        # Normalize Vietnamese accents → ASCII for robust matching against lexicon
        text_lower = self._normalize_vi(text.lower())
        scores: dict[str, float] = {e: 0.0 for e in EMOTION_CLASSES}

        for emotion, keywords in _VI_LEXICON.items():
            for kw in keywords:
                norm_kw = self._normalize_vi(kw)
                # Require at least 3 chars to avoid single-character false matches (ua, ma, lo...)
                if len(norm_kw) >= 3 and norm_kw in text_lower:
                    weight = 1.0 + min(1.0, len(kw.split()) * 0.15)
                    scores[emotion] += weight

        total = sum(scores.values())
        if total == 0:
            return {e: 0.0 for e in EMOTION_CLASSES}
        return {e: s / total for e, s in scores.items()}

    @staticmethod
    def _normalize_vi(text: str) -> str:
        """Remove Vietnamese diacritics for ASCII-matching fallback."""
        import unicodedata
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    def predict(self, text: str) -> list[dict[str, Any]]:
        """Predict emotion scores for a Vietnamese transcript.

        Args:
            text: Vietnamese transcript string.

        Returns:
            List of dicts with "label" and "score", sorted by score descending.
        """
        if not text or not text.strip():
            return [{"label": e, "score": 0.0} for e in EMOTION_CLASSES]

        # Component 1: PhoBERT zero-shot (60% weight)
        bert_scores = {e: 0.0 for e in EMOTION_CLASSES}
        if self._prototype_embeddings:
            text_vec = self._encode_text(text)
            if text_vec is not None:
                for emotion, proto_vec in self._prototype_embeddings.items():
                    bert_scores[emotion] = max(0.0, self._cosine_sim(text_vec, proto_vec))

        # Component 2: Lexicon scoring (40% weight)
        lex_scores = self._lexicon_scores(text)

        # Weighted combination
        BERT_WEIGHT = 0.6
        combined = {
            e: BERT_WEIGHT * bert_scores[e] + (1 - BERT_WEIGHT) * lex_scores[e]
            for e in EMOTION_CLASSES
        }

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {e: v / total for e, v in combined.items()}

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [{"label": label, "score": round(score, 4)} for label, score in ranked]


# Singleton instance
def _build_classifier() -> VietnameseTextEmotionClassifier:
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    return VietnameseTextEmotionClassifier(device=device)


text_emotion_classifier = _build_classifier()
