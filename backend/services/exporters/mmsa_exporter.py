"""MMSA-format dataset exporter.

Exports approved clips from the EDS SQLite database into a `.pkl` file
compatible with the MMSA DataLoader used by MulT training.

Output format:
    {
        "train": {
            "text":    np.ndarray (N_train, 50, 768), float32
            "audio":   np.ndarray (N_train, 50, 74),  float64
            "vision":  np.ndarray (N_train, 50, 35),  float64
            "id":      list[str]   (N_train,)
            "regression_labels":    np.ndarray (N_train,),     float64  # unused, all zeros
            "classification_labels": np.ndarray (N_train, 6),  float64  # multi-label [0,3]
        },
        "valid": { ... },
        "test":  { ... },
    }

Train/Val/Test split is done at the VIDEO level (not clip level) to prevent
data leakage — all clips from the same video go into the same split.

Emotion → index mapping (matching MulT output in notebook 05 and EDS emotion_analyzer):
    0: happy, 1: sad, 2: angry, 3: surprise, 4: disgust, 5: fear
    (6 emotions; "neutral" is excluded — clips labeled neutral get dominant emotion or 0 intensity)
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from backend.database.models import Clip, Video, Feature

# 6 emotions — ORDER MUST MATCH MulT output indices (notebook 05):
#   0: happy, 1: sad, 2: angry, 3: surprise, 4: disgust, 5: fear
EMOTION_CLASSES = ["happy", "sad", "angry", "surprise", "disgust", "fear"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_CLASSES)}
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, valid, test


class MMSAExporter:
    def __init__(self, db: Session):
        self.db = db

    def export(
        self,
        output_path: str,
        feature_dir: str | Path,
        random_seed: int = 42,
        label_source: str = "ai_auto",
        require_aligned: bool = True,
    ) -> dict[str, Any]:
        """Export all approved, aligned clips to MMSA-format .pkl.

        Args:
            output_path: Destination .pkl file path.
            feature_dir: Base directory where feature .npy files are stored.
            random_seed: Seed for reproducible train/val/test split.
            label_source: Filter clips by label_source ("ai_auto", "human_verified", etc.).
            require_aligned: If True, only export clips that have aligned features in Feature table.

        Returns:
            Summary dict with counts per split.
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        feature_dir = Path(feature_dir)

        # ── Load all approved clips ──────────────────────────────────────────────
        query = self.db.query(Clip).filter(
            Clip.status.in_(["approved", "auto_approved"])
        )
        if require_aligned:
            # Only clips that have a Feature record with aligned=True
            aligned_clip_ids = [
                f.clip_id for f in self.db.query(Feature).filter(
                    Feature.aligned == True
                ).all()
            ]
            query = query.filter(Clip.id.in_(aligned_clip_ids))

        clips = query.all()
        if not clips:
            return {"error": "No approved clips found", "total": 0}

        # ── Group clips by video ────────────────────────────────────────────────
        video_ids = list({c.video_id for c in clips})
        random.shuffle(video_ids)

        n_train = int(len(video_ids) * SPLIT_RATIOS[0])
        n_val = int(len(video_ids) * SPLIT_RATIOS[1])
        train_videos = set(video_ids[:n_train])
        val_videos = set(video_ids[n_train:n_train + n_val])
        test_videos = set(video_ids[n_train + n_val:])

        # ── Build classification labels matrix ─────────────────────────────────
        # Multi-label format: (N, 6) with values in [0, 3]
        def build_classification_labels(clip: Clip) -> np.ndarray:
            """Convert emotion info to multi-label [0,3] array for 6 emotions."""
            labels = np.zeros(6, dtype=np.float64)

            # Use per-model scores if available, else predicted_emotion
            all_scores = clip.all_scores or {}
            per_model = clip.per_model_scores or {}

            for emotion, idx in EMOTION_TO_IDX.items():
                # Priority: explicit score > all_scores > default
                score = all_scores.get(emotion)
                if score is None and per_model:
                    # Try to extract from per-model scores
                    for model_data in per_model.values():
                        if isinstance(model_data, dict):
                            if model_data.get("emotion") == emotion:
                                score = model_data.get("confidence", 0.0)
                            elif emotion in model_data:
                                score = model_data[emotion]

                if score is not None:
                    # Scale confidence [0,1] → intensity [0,3]
                    intensity = min(3.0, float(score) * 3.0)
                    labels[idx] = intensity
                else:
                    # Fallback: check predicted_emotion match
                    if clip.predicted_emotion and clip.predicted_emotion.lower().startswith(emotion):
                        labels[idx] = 1.0

            return labels

        def load_features(clip_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
            """Load text, audio, vision features for a clip. Returns None if any missing."""
            text_path = feature_dir / f"{clip_id}_text_aligned.npy"
            audio_path = feature_dir / f"{clip_id}_audio_aligned.npy"
            vision_path = feature_dir / f"{clip_id}_vision_aligned.npy"

            if not (text_path.exists() and audio_path.exists() and vision_path.exists()):
                return None

            text = np.load(str(text_path))   # (50, 768)
            audio = np.load(str(audio_path))  # (50, 74)
            vision = np.load(str(vision_path))  # (50, 35)

            # Validate shapes
            if text.shape != (50, 768) or audio.shape != (50, 74) or vision.shape != (50, 35):
                return None

            return text.astype(np.float32), audio.astype(np.float64), vision.astype(np.float64)

        # ── Assemble data per split ────────────────────────────────────────────
        def build_split(clips_for_split: list[Clip]) -> dict[str, Any]:
            text_list, audio_list, vision_list = [], [], []
            ids, reg_labels, cls_labels = [], [], []

            for clip in clips_for_split:
                feats = load_features(clip.id)
                if feats is None:
                    continue
                text_f, audio_f, vision_f = feats

                text_list.append(text_f)
                audio_list.append(audio_f)
                vision_list.append(vision_f)
                ids.append(clip.id)
                reg_labels.append(0.0)  # placeholder — not used
                cls_labels.append(build_classification_labels(clip))

            return {
                "text": np.stack(text_list) if text_list else np.zeros((0, 50, 768)),
                "audio": np.stack(audio_list) if audio_list else np.zeros((0, 50, 74)),
                "vision": np.stack(vision_list) if vision_list else np.zeros((0, 50, 35)),
                "id": ids,
                "regression_labels": np.array(reg_labels, dtype=np.float64),
                "classification_labels": np.stack(cls_labels) if cls_labels else np.zeros((0, 6)),
            }

        train_clips = [c for c in clips if c.video_id in train_videos]
        val_clips = [c for c in clips if c.video_id in val_videos]
        test_clips = [c for c in clips if c.video_id in test_videos]

        dataset = {
            "train": build_split(train_clips),
            "valid": build_split(val_clips),
            "test": build_split(test_clips),
        }

        # ── Write .pkl ────────────────────────────────────────────────────────
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "wb") as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        # ── Summary ───────────────────────────────────────────────────────────
        summary = {
            "output_path": str(output_path.resolve()),
            "total_clips": len(clips),
            "train_clips": len(train_clips),
            "valid_clips": len(val_clips),
            "test_clips": len(test_clips),
            "train_videos": len(train_videos),
            "valid_videos": len(val_videos),
            "test_videos": len(test_videos),
            "emotion_classes": EMOTION_CLASSES,
            "emotion_to_idx": EMOTION_TO_IDX,
        }
        return summary
