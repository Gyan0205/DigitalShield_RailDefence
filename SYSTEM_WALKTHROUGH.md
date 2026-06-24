# Digital Shield: Multi-Model Surveillance Overview

This document explains the unified workflow of the Digital Shield system, which combines CCTV computer vision with ticketing data intelligence to detect high-risk anomalies at Secunderabad Junction (SC).

---

### Step 1: CCTV Anomaly Detection (Visual Analysis)
The process begins with the CCTV Anomaly Detection model scanning video footage in real-time or from storage.
*   **What it does:** It analyzes video frames for suspicious behaviors (fighting, loitering, unauthorized access).
*   **The Output:** It generates a **CCTV Anomaly Score** (e.g., 0.85).
*   **Trigger:** If the score exceeds a certain threshold, the system proceeds to Step 2.

### Step 2: Metadata Extraction (Context Retrieval)
Once a visual anomaly is flagged, the system retrieves the situational context for that specific video.
*   **Source:** Each video file has a permanent `.json` sidecar file containing its metadata.
*   **Fields Retrieved:** Platform Number, Date, Time, and Day of the week.

### Step 3: Train Number Derivation (Railway Intelligence)
The system then identifies which train was at the platform during the anomaly.
*   **How it works:** It queries the `trains` table in the database using the **Platform Number**, **Time**, and **Day** from the metadata.
*   **The Result:** It identifies the specific **Train Number** (e.g., "12727 - Godavari Express").

### Step 4: Ticketing Anomaly Trigger (Secondary Verification)
Using the derived Train Number and the Date, the system triggers the **Tickets Database Anomaly Detection model**.
*   **What it does:** This model analyzes the booking patterns for that specific train on that specific date.
*   **The Output:** It generates a **Tickets Anomaly Score** based on suspicious booking activities (e.g., controller-dominated bookings, high-risk passenger clusters).

### Step 5: Final Risk Fusion (The Decision)
The system combines both scores to calculate the final risk level for security officers.
*   **The Logic:** A weighted average is calculated:
    *   **60% Weight:** CCTV Anomaly Score (Visual evidence)
    *   **40% Weight:** Tickets Anomaly Score (Data evidence)
*   **The Result:** A final **Risk Score** is sent to the RPF dashboard with a human-readable explanation of why the alert was triggered.

---

### Key Summary
1.  **CCTV** detects suspicious behavior visually.
2.  **Metadata** provides the "where" and "when".
3.  **Database** identifies the "which train".
4.  **Ticket Analysis** checks for suspicious booking patterns on that train.
5.  **Risk Fusion** combines visual and data intelligence for a highly accurate security alert.
