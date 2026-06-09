"""
Emotion Data Studio - dataset export worker.

Xuất dataset đa phương thức phục vụ huấn luyện sau này:
- labels.csv / labels.jsonl ở thư mục gốc
- metadata.json, dataset_card.md, quality_report.jsonl
- splits/*.jsonl
- full export: clips/, audio/, annotations/face_detections/
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal


class ExportWorker(QThread):
    progress_updated = Signal(int, str)
    export_finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        export_dir: str,
        export_format: str = "compact",
        approved_only: bool = True,
        auto_split: bool = True,
        stratified: bool = True,
        video_id: str | None = None,
    ):
        super().__init__()
        self.export_dir = Path(export_dir)
        self.export_format = export_format
        self.approved_only = approved_only
        self.auto_split = auto_split
        self.stratified = stratified
        self.video_id = video_id
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.progress_updated.emit(5, "Đang tải danh sách clip từ cơ sở dữ liệu...")
            from backend.database.local_db import get_session
            from backend.database.models import Clip

            session = get_session()
            try:
                query = session.query(Clip)
                if self.video_id:
                    query = query.filter(Clip.video_id == self.video_id)
                if self.approved_only:
                    query = query.filter(Clip.status.in_(["approved", "auto_approved"]))
                clips = query.order_by(Clip.video_id.asc(), Clip.clip_index.asc()).all()
                records = [self._clip_to_record(clip) for clip in clips]
            finally:
                session.close()

            valid_records, rejected_records = self._quality_gate(records)
            if not valid_records:
                raise RuntimeError(
                    "Không có clip hợp lệ để export. Hãy duyệt clip, kiểm tra nhãn và file media."
                )

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_dir = self.export_dir / f"emotion_dataset_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            self.progress_updated.emit(
                12,
                f"Quality gate hoàn tất: {len(valid_records)} clip hợp lệ, {len(rejected_records)} clip bị loại",
            )

            splits = self._split_records(valid_records)
            self._prepare_export_paths(output_dir, splits)
            self._write_core_metadata(output_dir, valid_records, rejected_records, splits)

            if self.export_format == "labels_only":
                self._export_labels_only(output_dir, splits)
            elif self.export_format == "full":
                self._export_full(output_dir, splits)
            else:
                self._export_compact(output_dir, splits)

            self._write_dataset_card(output_dir, valid_records, rejected_records, splits)
            self.progress_updated.emit(100, "Export hoàn tất")
            self.export_finished.emit(str(output_dir))
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def _clip_to_record(self, clip) -> dict:
        per_model = clip.per_model_scores or {}
        segment = per_model.get("segment") or {}
        face_info = per_model.get("face_extraction") or {}
        audio_info = per_model.get("audio_features") or {}
        final_emotion = clip.user_emotion or clip.ai_emotion
        label_source = self._label_source(clip)
        clip_path = Path(clip.clip_path) if clip.clip_path else None
        return {
            "clip_id": clip.id,
            "video_id": clip.video_id,
            "clip_index": clip.clip_index,
            "clip_path": str(clip_path) if clip_path else "",
            "file_name": clip_path.name if clip_path else "",
            "start_time": clip.start_time,
            "end_time": clip.end_time,
            "duration": clip.duration,
            "ai_emotion": clip.ai_emotion,
            "ai_confidence": clip.ai_confidence,
            "user_emotion": clip.user_emotion,
            "final_emotion": final_emotion,
            "label": final_emotion,
            "label_source": label_source,
            "quality_score": clip.quality_score,
            "transcript": clip.transcript or "",
            "status": clip.status,
            "has_incongruity": bool(clip.has_incongruity),
            "reviewer_notes": clip.reviewer_notes or "",
            "all_scores": clip.all_scores or {},
            "per_model_scores": per_model,
            "face_coverage": segment.get("face_coverage"),
            "speech_coverage": segment.get("speech_coverage"),
            "num_faces_avg": segment.get("num_faces_avg"),
            "segment_source": segment.get("source"),
            "segment_quality_hint": segment.get("quality_hint"),
            "cut_reason": segment.get("cut_reason"),
            "face_detections_path": face_info.get("detections_path", ""),
            "track_count": face_info.get("track_count", 0),
            "face_detector": face_info.get("detector", ""),
            "audio_path": audio_info.get("audio_path", ""),
            "audio_clarity": audio_info.get("audio_clarity"),
            "has_speech_energy": audio_info.get("has_speech_energy"),
        }

    @staticmethod
    def _label_source(clip) -> str:
        if clip.user_emotion:
            return "human_verified" if clip.user_emotion == clip.ai_emotion else "human_corrected"
        if clip.status == "auto_approved":
            return "ai_auto"
        return "ai_suggested"

    def _quality_gate(self, records: list[dict]) -> tuple[list[dict], list[dict]]:
        valid = []
        rejected = []
        seen = set()
        for record in records:
            reasons = []
            if not record.get("final_emotion"):
                reasons.append("missing_label")
            path = record.get("clip_path")
            if not path or not Path(path).exists():
                reasons.append("missing_file")
            if record.get("duration") is None or float(record.get("duration") or 0) <= 0:
                reasons.append("invalid_duration")
            if record.get("clip_id") in seen:
                reasons.append("duplicate_clip_id")
            seen.add(record.get("clip_id"))

            if reasons:
                bad = dict(record)
                bad["rejection_reasons"] = reasons
                rejected.append(bad)
            else:
                valid.append(record)
        return valid, rejected

    def _split_records(self, records: list[dict]) -> dict[str, list[dict]]:
        if not self.auto_split or len(records) < 3:
            return {"all": records}

        rng = random.Random(42)
        if self.stratified:
            grouped: dict[str, list[dict]] = defaultdict(list)
            for record in records:
                grouped[record["final_emotion"]].append(record)
            splits = {"train": [], "val": [], "test": []}
            for emotion_records in grouped.values():
                rng.shuffle(emotion_records)
                n = len(emotion_records)
                train_end = max(1, int(n * 0.70))
                val_end = train_end + max(0, int(n * 0.15))
                splits["train"].extend(emotion_records[:train_end])
                splits["val"].extend(emotion_records[train_end:val_end])
                splits["test"].extend(emotion_records[val_end:])
        else:
            shuffled = list(records)
            rng.shuffle(shuffled)
            n = len(shuffled)
            train_end = int(n * 0.70)
            val_end = train_end + int(n * 0.15)
            splits = {
                "train": shuffled[:train_end],
                "val": shuffled[train_end:val_end],
                "test": shuffled[val_end:],
            }
        return {name: values for name, values in splits.items() if values}

    def _prepare_export_paths(self, output_dir: Path, splits: dict[str, list[dict]]):
        for split_name, records in splits.items():
            for record in records:
                emotion = self._safe_name(record["final_emotion"])
                clip_src = Path(record["clip_path"])
                record["exported_clip_path"] = str(Path("clips") / split_name / emotion / f"{record['clip_id']}{clip_src.suffix.lower()}")
                if record.get("audio_path"):
                    audio_src = Path(record["audio_path"])
                    record["exported_audio_path"] = str(Path("audio") / split_name / emotion / f"{record['clip_id']}{audio_src.suffix.lower()}")
                if record.get("face_detections_path"):
                    record["exported_face_detections_path"] = str(Path("annotations") / "face_detections" / f"{record['clip_id']}.json")
                record["split"] = split_name
                record["quality_flags"] = self._quality_flags(record)

    def _quality_flags(self, record: dict) -> list[str]:
        flags = []
        if record.get("label_source", "").startswith("human"):
            flags.append("human_verified")
        if float(record.get("face_coverage") or 0) >= 0.5 or int(record.get("track_count") or 0) > 0:
            flags.append("has_face")
        if float(record.get("speech_coverage") or 0) >= 0.2 or record.get("transcript"):
            flags.append("has_dialogue")
        if record.get("has_incongruity"):
            flags.append("incongruity")
        if record.get("segment_quality_hint"):
            flags.append(f"segment_{record['segment_quality_hint']}")
        return flags

    def _write_core_metadata(self, output_dir: Path, valid: list[dict], rejected: list[dict], splits: dict[str, list[dict]]):
        counts = Counter(record["final_emotion"] for record in valid)
        label_sources = Counter(record.get("label_source") for record in valid)
        durations = [float(record.get("duration") or 0) for record in valid]
        metadata = {
            "created_at_utc": datetime.utcnow().isoformat(),
            "format": self.export_format,
            "approved_only": self.approved_only,
            "auto_split": self.auto_split,
            "stratified": self.stratified,
            "total_valid_clips": len(valid),
            "total_rejected_clips": len(rejected),
            "emotion_counts": dict(counts),
            "label_source_counts": dict(label_sources),
            "splits": {name: len(items) for name, items in splits.items()},
            "duration_sec": {
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "avg": sum(durations) / len(durations) if durations else 0,
            },
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        self._write_jsonl(output_dir / "quality_report.jsonl", self._quality_report_records(valid, rejected))
        if rejected:
            self._write_csv(output_dir / "quality_gate_rejected.csv", rejected)
        all_records = [record for items in splits.values() for record in items]
        self._write_csv(output_dir / "labels.csv", all_records)
        self._write_jsonl(output_dir / "labels.jsonl", all_records)
        transcripts = [self._transcript_record(r) for r in all_records if r.get("transcript")]
        if transcripts:
            self._write_jsonl(output_dir / "transcripts.jsonl", transcripts)

    def _quality_report_records(self, valid: list[dict], rejected: list[dict]) -> list[dict]:
        rows = []
        for record in valid:
            rows.append({
                "clip_id": record.get("clip_id"),
                "status": "accepted",
                "label": record.get("final_emotion"),
                "quality_score": record.get("quality_score"),
                "face_coverage": record.get("face_coverage"),
                "speech_coverage": record.get("speech_coverage"),
                "track_count": record.get("track_count"),
                "flags": record.get("quality_flags", []),
            })
        for record in rejected:
            rows.append({
                "clip_id": record.get("clip_id"),
                "status": "rejected",
                "reasons": record.get("rejection_reasons", []),
            })
        return rows

    @staticmethod
    def _transcript_record(record: dict) -> dict:
        return {
            "clip_id": record.get("clip_id"),
            "video_id": record.get("video_id"),
            "split": record.get("split"),
            "label": record.get("final_emotion"),
            "transcript": record.get("transcript"),
        }

    def _export_labels_only(self, output_dir: Path, splits: dict[str, list[dict]]):
        self.progress_updated.emit(35, "Đã ghi labels.csv và labels.jsonl")
        self._write_split_files(output_dir, splits)

    def _export_compact(self, output_dir: Path, splits: dict[str, list[dict]]):
        self.progress_updated.emit(25, "Đang ghi metadata theo split...")
        self._write_split_files(output_dir, splits)

    def _write_split_files(self, output_dir: Path, splits: dict[str, list[dict]]):
        splits_dir = output_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        total = max(1, sum(len(items) for items in splits.values()))
        done = 0
        for split_name, records in splits.items():
            self._write_csv(splits_dir / f"{split_name}.csv", records)
            self._write_jsonl(splits_dir / f"{split_name}.jsonl", records)
            done += len(records)
            self.progress_updated.emit(25 + int(60 * done / total), f"Đã ghi split {split_name}: {len(records)} clip")

    def _export_full(self, output_dir: Path, splits: dict[str, list[dict]]):
        self.progress_updated.emit(20, "Đang copy clip, audio và annotation...")
        total = max(1, sum(len(items) for items in splits.values()))
        done = 0
        self._write_split_files(output_dir, splits)
        for split_name, records in splits.items():
            for record in records:
                if self._cancelled:
                    raise RuntimeError("Đã hủy export")
                self._copy_optional(record.get("clip_path"), output_dir / record["exported_clip_path"])
                self._copy_optional(record.get("audio_path"), output_dir / record.get("exported_audio_path", ""))
                self._copy_optional(record.get("face_detections_path"), output_dir / record.get("exported_face_detections_path", ""))
                done += 1
                if done % 5 == 0 or done == total:
                    self.progress_updated.emit(30 + int(60 * done / total), f"Đã copy {done}/{total} mẫu")

    @staticmethod
    def _copy_optional(src: str | None, dst: Path):
        if not src or not str(dst):
            return
        source = Path(src)
        if not source.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dst)

    def _write_csv(self, path: Path, records: list[dict]):
        fieldnames = [
            "clip_id", "video_id", "clip_index", "split", "clip_path", "exported_clip_path",
            "audio_path", "exported_audio_path", "face_detections_path", "exported_face_detections_path",
            "start_time", "end_time", "duration", "ai_emotion", "ai_confidence",
            "user_emotion", "final_emotion", "label", "label_source", "quality_score",
            "face_coverage", "speech_coverage", "num_faces_avg", "track_count", "segment_source",
            "segment_quality_hint", "cut_reason", "transcript", "status", "has_incongruity",
            "reviewer_notes", "quality_flags", "rejection_reasons",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                row = dict(record)
                for key in ("rejection_reasons", "quality_flags"):
                    if isinstance(row.get(key), list):
                        row[key] = ";".join(str(v) for v in row[key])
                writer.writerow(row)

    def _write_jsonl(self, path: Path, records: list[dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_dataset_card(self, output_dir: Path, valid: list[dict], rejected: list[dict], splits: dict[str, list[dict]]):
        counts = Counter(record["final_emotion"] for record in valid)
        label_sources = Counter(record.get("label_source") for record in valid)
        avg_duration = sum(float(r.get("duration") or 0) for r in valid) / len(valid) if valid else 0
        lines = [
            "# Emotion Dataset Export",
            "",
            "## Tổng quan",
            f"- Thời điểm tạo UTC: {datetime.utcnow().isoformat()}",
            f"- Định dạng export: {self.export_format}",
            f"- Clip hợp lệ: {len(valid)}",
            f"- Clip bị loại bởi quality gate: {len(rejected)}",
            f"- Thời lượng trung bình: {avg_duration:.2f}s",
            "",
            "## Phân chia tập dữ liệu",
        ]
        for name, items in splits.items():
            lines.append(f"- {name}: {len(items)}")
        lines.extend(["", "## Phân bố cảm xúc"])
        for emotion, count in sorted(counts.items()):
            lines.append(f"- {emotion}: {count}")
        lines.extend(["", "## Nguồn nhãn"])
        for source, count in sorted(label_sources.items()):
            lines.append(f"- {source}: {count}")
        lines.extend([
            "",
            "## Cấu trúc file chính",
            "- labels.csv / labels.jsonl: nhãn và metadata chính",
            "- splits/*.jsonl: metadata theo train/val/test hoặc all",
            "- quality_report.jsonl: báo cáo quality gate",
            "- transcripts.jsonl: lời thoại nếu có",
            "- annotations/face_detections/*.json: bbox khuôn mặt nếu export full",
            "",
            "## Lưu ý",
            "- Nhãn `human_verified` hoặc `human_corrected` nên được ưu tiên khi huấn luyện.",
            "- Nhãn `ai_auto` chỉ nên dùng sau khi kiểm tra quality report.",
        ])
        (output_dir / "dataset_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value or "unknown")
