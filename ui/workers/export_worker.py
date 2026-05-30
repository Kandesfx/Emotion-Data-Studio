"""
Emotion Data Studio — Export Worker (QThread)
==============================================
Runs dataset export in background thread.
"""

from PySide6.QtCore import QThread, Signal
import os
import json
import csv


class ExportWorker(QThread):
    """Background worker for exporting dataset"""

    progress_updated = Signal(int, str)     # percentage, message
    export_finished = Signal(str)            # output path
    error_occurred = Signal(str)             # error message

    def __init__(self, export_dir: str, export_format: str = "compact",
                 approved_only: bool = True, auto_split: bool = True,
                 stratified: bool = True):
        super().__init__()
        self.export_dir = export_dir
        self.export_format = export_format
        self.approved_only = approved_only
        self.auto_split = auto_split
        self.stratified = stratified

    def run(self):
        """Execute export in background thread"""
        try:
            self.progress_updated.emit(5, "Loading clips from database...")

            from backend.database.local_db import get_session
            from backend.database.models import Clip, Video

            session = get_session()
            try:
                # Query clips
                query = session.query(Clip)
                if self.approved_only:
                    query = query.filter(
                        Clip.status.in_(['approved', 'auto_approved'])
                    )
                clips = query.all()

                if not clips:
                    self.error_occurred.emit("No clips to export")
                    return

                self.progress_updated.emit(10, f"Found {len(clips)} clips to export")

                # Create output directory
                output_dir = os.path.join(self.export_dir, "emotion_dataset")
                os.makedirs(output_dir, exist_ok=True)

                if self.export_format == "labels_only":
                    self._export_labels_csv(clips, output_dir)
                elif self.export_format == "compact":
                    self._export_compact(clips, output_dir)
                else:
                    self._export_full(clips, output_dir)

                self.progress_updated.emit(95, "Generating metadata...")

                # Write metadata
                metadata = {
                    "total_clips": len(clips),
                    "format": self.export_format,
                    "approved_only": self.approved_only,
                    "auto_split": self.auto_split,
                    "emotions": list(set(
                        c.human_emotion or c.ai_emotion
                        for c in clips if c.human_emotion or c.ai_emotion
                    )),
                }
                with open(os.path.join(output_dir, "metadata.json"), "w",
                           encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                self.progress_updated.emit(100, "Export complete!")
                self.export_finished.emit(output_dir)

            finally:
                session.close()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _export_labels_csv(self, clips, output_dir):
        """Export labels only as CSV"""
        self.progress_updated.emit(30, "Writing labels CSV...")

        csv_path = os.path.join(output_dir, "labels.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "clip_id", "video_id", "start_time", "end_time", "duration",
                "ai_emotion", "ai_confidence", "human_emotion", "final_emotion",
                "quality_score", "transcript", "status"
            ])

            for i, clip in enumerate(clips):
                final = clip.human_emotion or clip.ai_emotion
                writer.writerow([
                    clip.id, clip.video_id, clip.start_time, clip.end_time,
                    clip.duration, clip.ai_emotion, clip.ai_confidence,
                    clip.human_emotion, final, clip.quality_score,
                    clip.transcript, clip.status
                ])

                if i % 50 == 0:
                    pct = 30 + int(60 * i / len(clips))
                    self.progress_updated.emit(pct, f"Writing clip {i+1}/{len(clips)}")

    def _export_compact(self, clips, output_dir):
        """Export compact format (metadata + file references)"""
        self.progress_updated.emit(20, "Exporting compact format...")

        # Create train/val/test splits if requested
        if self.auto_split:
            from sklearn.model_selection import train_test_split

            labels = [c.human_emotion or c.ai_emotion or "neutral" for c in clips]

            try:
                train_clips, temp_clips, train_labels, temp_labels = train_test_split(
                    clips, labels, test_size=0.3,
                    stratify=labels if self.stratified else None,
                    random_state=42
                )
                val_clips, test_clips, _, _ = train_test_split(
                    temp_clips, temp_labels, test_size=0.5,
                    stratify=temp_labels if self.stratified else None,
                    random_state=42
                )
            except ValueError:
                # Not enough samples for stratification
                train_clips, temp_clips = train_test_split(
                    clips, test_size=0.3, random_state=42
                )
                val_clips, test_clips = train_test_split(
                    temp_clips, test_size=0.5, random_state=42
                )

            splits = {"train": train_clips, "val": val_clips, "test": test_clips}
        else:
            splits = {"all": clips}

        total = len(clips)
        processed = 0

        for split_name, split_clips in splits.items():
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            # Write split CSV
            csv_path = os.path.join(split_dir, f"{split_name}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "clip_id", "clip_path", "emotion", "confidence",
                    "quality_score", "transcript"
                ])

                for clip in split_clips:
                    final_emotion = clip.human_emotion or clip.ai_emotion or "neutral"
                    writer.writerow([
                        clip.id, clip.clip_path, final_emotion,
                        clip.ai_confidence, clip.quality_score, clip.transcript
                    ])
                    processed += 1

                    if processed % 20 == 0:
                        pct = 20 + int(70 * processed / total)
                        self.progress_updated.emit(
                            pct, f"Exporting {split_name}: {processed}/{total}"
                        )

    def _export_full(self, clips, output_dir):
        """Export full format (copy files + metadata)"""
        self.progress_updated.emit(20, "Exporting full format (this may take a while)...")
        # Same as compact but also copies video/audio files
        self._export_compact(clips, output_dir)
        # TODO: Add file copying logic for full export
