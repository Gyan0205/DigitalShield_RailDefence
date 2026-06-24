# Digital Shield: Developer Session Log
**Session Date:** 2026-05-15

## 1. Project Pivot: CCTV-Only Surveillance
The primary goal of this session was to finalize the transition from a hybrid passenger/CCTV system to a **pure CCTV anomaly detection system** for Secunderabad Junction (SC).

### Major Code Cleanup
- **Deleted**: `backend/models/database.py` (legacy passenger models).
- **Modified**: `backend/database.py` to remove `PassengerModel` and all passenger-related stats methods.
- **Modified**: `deploy/init.sql` to remove the `passengers` table and its indexes.
- **Modified**: `backend/services/xai_engine.py` to remove "Passenger Narrowing" and "Booking Anomaly Scoring" stages.
- **Audit Clean**: Updated `final_audit.py` and test scripts to remove all references to passengers/booking.

## 2. Metadata System Refactor
We moved the metadata system from a live database dependency to a local, permanent context system.

### Operational Context Extraction
- **Command Run**: Connected to Supabase and extracted 110 real timetable patterns (Platform, Time, Day) for SC.
- **Storage**: Saved to `backend/data/sc_context.json`.
- **Reasoning**: To allow the system to work offline and with zero database overhead.

### Permanent Metadata Sidecars
- **Logic**: The `MetadataPipeline` now assigns a context to every video within the **Jan 1st – March 27th, 2026** window.
- **Storage**: Each video now gets a permanent `.json` sidecar file (e.g., `camera_1.mp4` -> `camera_1.json`).
- **Fields Retained**:
  1. Platform Number
  2. Date
  3. Time
  4. Day
  5. Station (Secunderabad Junction)
- **Fields Removed**: Train Numbers and Anomaly Types (these are now treated as AI *outputs*, not inputs).

## 3. Intelligence Pipeline Rebalancing
- **XAI Confidence Weights**: Rebalanced to prioritize CCTV signals:
  - **CCTV Anomaly**: 50%
  - **Train Match**: 30%
  - **Coach Estimation**: 20%
- **Inference Logic**: The system now infers the train number at runtime by cross-referencing the video's Time/Platform against the timetable, rather than having it hardcoded in the video metadata.

## 4. Current Project State
- **Station Lockdown**: Fixed to Secunderabad (SC).
- **Database Dependency**: High-level alerts/metrics still use DB, but metadata labeling is now 100% local.
- **Environment Note**: Final audit passed all logic checks. Minor failures remain only for missing `cv2` (OpenCV) system dependencies on the host machine.

---

## 5. Session Date: 2026-05-17 — Multi-Model ML Fusion & Local Self-Containment

### Multi-Model Hybrid CCTV-Ticket Booking Fusion
- **Implemented 60-40 Hybrid Fusion**: Overhauled `fuse_event` in `backend/services/fusion_engine.py` to trigger the 60-40 multi-model fusion algorithm whenever a CCTV anomaly score is $\ge 60\%$:
  $$\text{Final Fused Score} = (0.60 \times \text{CCTV Score}) + (0.40 \times \text{Tickets Score})$$
- **Supabase Timetable Lookup**: Resolved scheduled train details directly from the Supabase PostgreSQL `trains` table at runtime by cross-referencing Platform, Time, and Travel Day.
- **Auto-Provisioned Production Database**: Executed SQLAlchemy table-provisioning to automatically create and initialize the missing `alerts`, `detections`, `audit_log`, and `train_schedules` tables directly inside the live AWS Supabase instance.
- **Fail-Safe Robustness**:
  - URL-encoded Supabase database passwords containing special characters (like `@`) to ensure connection stability.
  - Implemented dynamic headless imports catching on `cv2` (OpenCV) inside `coach_ocr.py` and `bogie_mapper.py` to guarantee crash-free backend startup.

### Self-Contained Repository Restructuring
- **Ticket Model Local Placement**: Copied the entire `DigitalShield2` ML passenger booking risk engine to [ticket_model](file:///c:/Users/Lenovo/Downloads/DigitalShield/DigitalShield/ticket_model), containing all necessary source code files (`risk_detection_engine.py`, `dashboard.py`, and `src/`).
- **Independence Achieved**: The `DigitalShield` CCTV project is now completely independent and self-contained, no longer requiring external folder traversals or operations on `DigitalShield2`.

