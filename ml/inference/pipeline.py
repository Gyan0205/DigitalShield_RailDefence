"""
Digital Shield Rail Defense — Unified Inference Pipeline
==========================================================
End-to-end CCTV anomaly detection pipeline that orchestrates:
  YOLOv8 Detection → DeepSORT Tracking → Pose Estimation →
  Behavioral Analysis → Anomaly Classification → Alert Generation

Supports:
  - Single video file processing
  - Real-time RTSP/webcam streams
  - Batch video processing
  - Configurable module selection
  - Structured output with timestamps
"""

import cv2
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger("inference_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s")


@dataclass
class PipelineConfig:
    """Inference pipeline configuration."""
    # Module toggles
    enable_detection: bool = True
    enable_tracking: bool = True
    enable_pose: bool = True
    enable_behavior: bool = True
    enable_classification: bool = True
    # Detection
    yolo_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    # Tracking
    max_age: int = 30
    # Processing
    frame_skip: int = 1
    max_frames: Optional[int] = None
    # Output
    save_annotated_video: bool = False
    save_results_json: bool = True
    output_dir: str = "output"


@dataclass
class FrameResult:
    """Complete analysis result for a single frame."""
    frame_idx: int
    timestamp: float
    person_count: int = 0
    track_count: int = 0
    anomalies: List[Dict] = field(default_factory=list)
    detections: List[Dict] = field(default_factory=list)
    tracks: List[Dict] = field(default_factory=list)
    has_anomaly: bool = False
    max_risk_level: str = "NORMAL"

    def to_dict(self) -> Dict:
        return asdict(self)


class InferencePipeline:
    """
    Unified inference pipeline orchestrating all detection modules.

    Usage:
        pipeline = InferencePipeline()
        results = pipeline.process_video("surveillance.mp4")
        # Or for real-time:
        pipeline.process_stream("rtsp://camera_url")
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.detector = None
        self.tracker = None
        self.pose_estimator = None
        self.behavior_analyzer = None
        self.anomaly_classifier = None
        self._initialized = False
        self._results: List[FrameResult] = []

    def initialize(self):
        """Lazy-load all pipeline modules."""
        if self._initialized:
            return

        logger.info("Initializing inference pipeline...")

        if self.config.enable_detection:
            from ml.models.yolo_detector import YOLODetector
            self.detector = YOLODetector(
                model_variant=self.config.yolo_model,
                confidence_threshold=self.config.confidence_threshold,
            )
            logger.info("  ✓ YOLOv8 detector loaded")

        if self.config.enable_tracking:
            from ml.models.deepsort_tracker import DeepSORTTracker
            self.tracker = DeepSORTTracker(max_age=self.config.max_age)
            logger.info("  ✓ DeepSORT tracker loaded")

        if self.config.enable_pose:
            try:
                from ml.models.pose_estimator import PoseEstimator
                self.pose_estimator = PoseEstimator()
                logger.info("  ✓ Pose estimator loaded")
            except Exception as e:
                logger.warning(f"  ⚠ Pose estimator unavailable: {e}")
                self.config.enable_pose = False

        if self.config.enable_behavior:
            from ml.models.behavior_analyzer import BehaviorAnalyzer
            self.behavior_analyzer = BehaviorAnalyzer()
            logger.info("  ✓ Behavior analyzer loaded")

        if self.config.enable_classification:
            from ml.models.anomaly_classifier import AnomalyClassifier
            self.anomaly_classifier = AnomalyClassifier()
            logger.info("  ✓ Anomaly classifier loaded")

        self._initialized = True
        logger.info("Pipeline initialized successfully")

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0,
                      timestamp: float = 0.0) -> FrameResult:
        """
        Process a single frame through the full pipeline.

        Returns:
            FrameResult with detections, tracks, and anomalies
        """
        if not self._initialized:
            self.initialize()

        result = FrameResult(frame_idx=frame_idx, timestamp=timestamp)

        # Stage 1: Person Detection
        frame_detections = None
        if self.detector:
            frame_detections = self.detector.detect(frame)
            result.person_count = frame_detections.person_count
            result.detections = [d.to_dict() for d in frame_detections.detections]

        # Stage 2: Multi-Person Tracking
        tracks = []
        if self.tracker and frame_detections:
            tracks = self.tracker.update(frame_detections, frame, frame_idx)
            result.track_count = len(tracks)
            result.tracks = [t.to_dict() for t in tracks]

        # Stage 3: Pose Estimation
        poses = []
        if self.pose_estimator and result.person_count > 0:
            try:
                poses = self.pose_estimator.estimate(frame)
            except Exception:
                pass

        # Stage 4: Behavioral Analysis
        if self.behavior_analyzer and tracks:
            pairwise = self.tracker.get_pairwise_distances(tracks) if self.tracker else None
            anomalies = self.behavior_analyzer.analyze_frame(
                tracks=tracks,
                poses=poses if poses else None,
                pairwise_distances=pairwise,
                frame_idx=frame_idx,
                timestamp=timestamp,
                frame_shape=frame.shape[:2],
            )

            for anomaly in anomalies:
                if anomaly.is_anomalous:
                    result.anomalies.append(anomaly.to_dict())
                    result.has_anomaly = True

            # Determine max risk level
            risk_priority = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
            for a in anomalies:
                if risk_priority.get(a.risk_level, 0) > risk_priority.get(result.max_risk_level, 0):
                    result.max_risk_level = a.risk_level

        return result

    def process_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Process an entire video file through the pipeline.

        Args:
            video_path: Path to input video
            output_dir: Directory for output files

        Returns:
            Complete analysis report
        """
        if not self._initialized:
            self.initialize()

        video_path = Path(video_path)
        out_dir = Path(output_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Processing: {video_path.name} ({total_frames} frames, {fps:.1f} FPS, {width}x{height})")

        # Video writer for annotated output
        writer = None
        if self.config.save_annotated_video:
            out_path = out_dir / f"{video_path.stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        self._results = []
        frame_idx = 0
        processed = 0
        anomaly_frames = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if self.config.max_frames and processed >= self.config.max_frames:
                break

            if frame_idx % self.config.frame_skip == 0:
                timestamp = frame_idx / fps if fps > 0 else 0
                result = self.process_frame(frame, frame_idx, timestamp)
                self._results.append(result)
                processed += 1

                if result.has_anomaly:
                    anomaly_frames += 1
                    for a in result.anomalies:
                        logger.warning(
                            f"  ANOMALY @ frame {frame_idx}: {a['anomaly_type']} "
                            f"({a['risk_level']}, conf={a['confidence']:.2f})"
                        )

                # Write annotated frame
                if writer and self.tracker:
                    tracks = self.tracker.get_all_tracks()
                    active = [t for t in tracks.values() if t.is_confirmed]
                    annotated = self.tracker.draw_tracks(frame.copy(), active)
                    if self.pose_estimator and self.config.enable_pose:
                        try:
                            poses = self.pose_estimator.estimate(frame)
                            annotated = self.pose_estimator.draw_poses(annotated, poses)
                        except Exception:
                            pass
                    # Draw anomaly alerts
                    if result.has_anomaly:
                        color = (0, 0, 255) if result.max_risk_level in ("HIGH", "CRITICAL") else (0, 165, 255)
                        cv2.putText(annotated, f"ALERT: {result.max_risk_level}", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    writer.write(annotated)

                if processed % 50 == 0:
                    elapsed = time.time() - start_time
                    fps_actual = processed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"  Frame {frame_idx}/{total_frames} | "
                        f"{processed} processed | {anomaly_frames} anomalies | "
                        f"{fps_actual:.1f} FPS"
                    )

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        elapsed = time.time() - start_time

        # Generate report
        report = self._generate_report(video_path, processed, anomaly_frames, elapsed, fps)

        # Save results
        if self.config.save_results_json:
            results_path = out_dir / f"{video_path.stem}_results.json"
            with open(results_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Results saved: {results_path}")

        return report

    def process_stream(
        self,
        source: Union[str, int] = 0,
        display: bool = True,
        callback=None,
    ):
        """
        Process a live CCTV stream (RTSP, webcam, etc.).

        Args:
            source: RTSP URL, webcam index, or video path
            display: Show live visualization window
            callback: Optional callback(frame_idx, result) per frame
        """
        if not self._initialized:
            self.initialize()

        cap = cv2.VideoCapture(source if isinstance(source, str) else int(source))
        if not cap.isOpened():
            raise ValueError(f"Cannot open stream: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_idx = 0

        logger.info(f"Stream started: {source}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Stream ended or disconnected")
                    break

                if frame_idx % self.config.frame_skip == 0:
                    timestamp = frame_idx / fps
                    result = self.process_frame(frame, frame_idx, timestamp)

                    if callback:
                        callback(frame_idx, result)

                    if result.has_anomaly:
                        for a in result.anomalies:
                            logger.warning(
                                f"LIVE ALERT: {a['anomaly_type']} ({a['risk_level']})"
                            )

                    if display:
                        vis = frame.copy()
                        if self.tracker:
                            tracks = self.tracker.get_all_tracks()
                            active = [t for t in tracks.values() if t.is_confirmed]
                            vis = self.tracker.draw_tracks(vis, active)

                        if result.has_anomaly:
                            cv2.putText(vis, f"ALERT: {result.max_risk_level}",
                                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        cv2.imshow("Digital Shield — Live Surveillance", vis)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                frame_idx += 1

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            logger.info(f"Stream ended: {frame_idx} frames processed")

    def _generate_report(self, video_path, processed, anomaly_frames, elapsed, fps) -> Dict:
        """Generate comprehensive analysis report."""
        all_anomalies = []
        for r in self._results:
            all_anomalies.extend(r.anomalies)

        # Anomaly timeline
        timeline = []
        for r in self._results:
            if r.has_anomaly:
                timeline.append({
                    "frame_idx": r.frame_idx,
                    "timestamp": r.timestamp,
                    "risk_level": r.max_risk_level,
                    "anomalies": r.anomalies,
                })

        # Aggregate by type
        by_type = {}
        for a in all_anomalies:
            atype = a["anomaly_type"]
            by_type.setdefault(atype, {"count": 0, "max_confidence": 0})
            by_type[atype]["count"] += 1
            by_type[atype]["max_confidence"] = max(by_type[atype]["max_confidence"], a["confidence"])

        behavior_summary = {}
        if self.behavior_analyzer:
            behavior_summary = self.behavior_analyzer.get_anomaly_summary()

        return {
            "pipeline": "Digital Shield Rail Defense — CCTV Anomaly Detection",
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "video": {
                "path": str(video_path),
                "name": video_path.name,
                "fps": fps,
            },
            "processing": {
                "frames_processed": processed,
                "anomaly_frames": anomaly_frames,
                "anomaly_rate": f"{anomaly_frames / processed * 100:.1f}%" if processed > 0 else "0%",
                "elapsed_seconds": round(elapsed, 2),
                "processing_fps": round(processed / elapsed, 1) if elapsed > 0 else 0,
            },
            "anomalies": {
                "total": len(all_anomalies),
                "by_type": by_type,
                "critical": sum(1 for a in all_anomalies if a.get("risk_level") == "CRITICAL"),
                "high": sum(1 for a in all_anomalies if a.get("risk_level") == "HIGH"),
            },
            "timeline": timeline,
            "behavior_summary": behavior_summary,
        }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Digital Shield — CCTV Anomaly Detection Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Video path or RTSP URL")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose estimation")
    parser.add_argument("--stream", action="store_true", help="Real-time stream mode")
    parser.add_argument("--webcam", type=int, help="Webcam index for live mode")

    args = parser.parse_args()

    config = PipelineConfig(
        yolo_model="yolov8n.pt",
        confidence_threshold=args.confidence,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        save_annotated_video=args.save_video,
        enable_pose=not args.no_pose,
        output_dir=args.output,
    )

    pipeline = InferencePipeline(config)

    if args.stream or args.webcam is not None:
        source = args.webcam if args.webcam is not None else args.input
        pipeline.process_stream(source)
    else:
        report = pipeline.process_video(args.input, args.output)
        logger.info(f"\nTotal anomalies: {report['anomalies']['total']}")
        logger.info(f"Critical: {report['anomalies']['critical']}, High: {report['anomalies']['high']}")


if __name__ == "__main__":
    main()
