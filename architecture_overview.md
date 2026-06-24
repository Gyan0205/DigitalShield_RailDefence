# Digital Shield — System Overview

This document provides a high-level overview of the **Digital Shield** anomaly detection system in its current state (CCTV-only pipeline).

## Core Architecture

Digital Shield has been streamlined to focus exclusively on **CCTV-based visual anomaly detection**, combined with railway metadata. The ticket booking and passenger database intelligence systems have been removed to reduce complexity and privacy footprint.

The system relies on a **4-Source Fusion Engine** to generate high-confidence intelligence alerts for the Railway Protection Force (RPF).

### The 4 Intelligence Sources

1. **CCTV Anomaly Detection (Weight: 46%)**
   - The primary source of threat intelligence.
   - Uses YOLOv8 for person detection and DeepSORT for multi-object tracking.
   - Identifies behavioral anomalies in video streams (e.g., suspicious escorts, loitering).

2. **Train Intelligence (Weight: 23%)**
   - Correlates CCTV events with train schedules.
   - Infers which train is at the platform during the event using the `ScheduleDB`.

3. **Railway Metadata (Weight: 15%)**
   - Provides physical context using `CameraRegistry` and `StationDB`.
   - Resolves camera IDs to specific platforms and station zones (e.g., Entry, Mid, Exit).

4. **Coach Estimation (Weight: 15%)**
   - Estimates the likely train coach the subjects are near or interacting with.
   - Uses `BogieMapper` to map camera zones and person positions to coach layouts.

## The Fusion Pipeline

When a camera detects a suspicious event, the payload is sent to the Fusion Engine via `POST /api/detect-anomaly`. 

1. **Event Bus:** The event is published to an asynchronous event bus.
2. **Context Enrichment:** The engine queries the Train, Metadata, and Coach modules to build context around the raw CCTV event.
3. **Dempster-Shafer Combination:** The engine combines the confidence scores from the 4 sources. Agreement between sources boosts the overall fused confidence.
4. **Temporal Correlation:** Events happening at the same station/platform within a 5-minute window are clustered into a single incident.
5. **Severity Classification:** The fused confidence score is mapped to a severity level (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
6. **Alert Generation:** A finalized `FusedAlert` is pushed to the frontend via WebSockets and saved to the database.

## System Interfaces

- **Frontend:** A React/Vite dashboard providing live CCTV feeds, analytics, platform monitoring, and a stream of actionable alerts.
- **Backend:** A FastAPI application exposing 6 core REST endpoints for video upload, anomaly detection, metadata lookup, and alert retrieval.

## ML Pipeline

The ML pipeline (located in the `ml/` directory) can run in inference mode on video files. It outputs bounding boxes, tracks, and behavioral classification which are then fed into the backend API. In the absence of a GPU or Ultralytics, the system falls back to a simulated rule-based detection engine for demonstration purposes.
