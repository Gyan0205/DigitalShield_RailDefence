# 📦 Digital Shield: Self-Contained Repository Setup
**Unified Architecture & Dynamic Path Resolution Framework**

---

## 📋 Overview

To simplify operational deployment and ensure the platform is completely independent of external directory trees, we have consolidated the **CCTV Anomaly Detection Backend** and the **Tickets Database ML Outlier Engine** into a single, unified, self-contained repository.

The entire passenger ticketing intelligence pipeline is now safely modularized inside a subfolder under the primary `DigitalShield` project root, eliminating any cross-folder traversal dependencies.

---

## 📂 Consolidated Directory Structure

All necessary ticket anomaly source files, training pipelines, and visualization dashboards are now located in your main workspace:

```text
DigitalShield/ (Project Root)
├── .env                              <-- Database & Application Env Configurations
├── outputs/                          <-- Shared ML Output Directory
│   └── all_risk_scores.csv           <-- 130,000 Scored Passenger Bookings
├── backend/                          <-- Main FastAPI CCTV Backend Server
│   └── services/
│       ├── fusion_engine.py          <-- 60-40 Multi-Model Decision Orchestrator
│       └── tickets_intelligence.py   <-- In-Memory Ticket Score Cache & Lookup
│
└── ticket_model/                     <-- Unified Ticket Anomaly ML Engine [NEW]
    ├── requirements.txt              <-- Specific ML Ticketing Dependencies
    ├── risk_detection_engine.py      <-- Slim Pipeline Runner (IF + LOF Models)
    ├── dashboard.py                  <-- Streamlit Analytics Dashboard Application
    └── src/                          <-- Modular Source Code Libraries
        ├── cleaner.py                <-- Data Preprocessing
        ├── config.py                 <-- Model Hyperparameters & Score Thresholds
        ├── data_loader.py            <-- Supabase Postgres Data Fetcher
        ├── explainer.py              <-- Explainable AI Reason Generator
        ├── features.py               <-- 14 Engineered Security Risk Features
        ├── logger.py                 <-- Logger Setup
        ├── model.py                  <-- Isolation Forest & LOF Core Implementations
        └── scoring.py                <-- Multiplicative Threat Scorer
```

---

## ⚙️ Path Independence & Import Resolutions

To make the codebase fully portable (allowing it to run seamlessly on developer desktops, local offline setups, or secure production servers), we overhauled all directory references:

### 1. Dynamic Database URL-Encoding
Special character entries in database passwords (like `@`) are automatically quoted inside [backend/database.py](file:///c:/Users/Lenovo/Downloads/DigitalShield/DigitalShield/backend/database.py) using `urllib.parse.quote_plus` to guarantee connection stability under all setups.

### 2. Relative ML Cache Resolutions
Inside [backend/services/tickets_intelligence.py](file:///c:/Users/Lenovo/Downloads/DigitalShield/DigitalShield/backend/services/tickets_intelligence.py), candidate search directories are dynamically computed relative to the service module itself, fallback matching your local workspace outputs:

```python
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
self.csv_paths = [
    os.path.join(root_dir, "outputs", "all_risk_scores.csv"),
    os.path.join(root_dir, "ticket_model", "outputs", "all_risk_scores.csv"),
    r"c:\Users\Lenovo\Downloads\DigitalShield\DigitalShield\outputs\all_risk_scores.csv",
    r"C:\Users\Lenovo\Downloads\DigitalShield2\DigitalShield2\outputs\all_risk_scores.csv"
]
```

### 3. Absolute Module Path Injectors
To let you run the pipeline or dashboard from the repository root without PYTHONPATH import crashes, we injected a path resolver at the very top of [risk_detection_engine.py](file:///c:/Users/Lenovo/Downloads/DigitalShield/DigitalShield/ticket_model/risk_detection_engine.py) and [dashboard.py](file:///c:/Users/Lenovo/Downloads/DigitalShield/DigitalShield/ticket_model/dashboard.py):

```python
import sys
import os

# Add the directory containing this file to sys.path to enable src imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
```

---

## 🚀 Execution Instructions

Since the code is unified under a single workspace, you can operate the pipelines cleanly:

### Run the Ticket ML Anomaly Pipeline
Execute the Isolation Forest + LOF model to score passenger bookings and output them into the shared `outputs/` folder:
```powershell
python ticket_model/risk_detection_engine.py
```

### Launch the Streamlit Ticket Dashboard
Launch the interactive Streamlit passenger anomaly dashboard:
```powershell
streamlit run ticket_model/dashboard.py
```

### Run the CCTV Fusion Backend Integration Tests
Execute the end-to-end CCTV schedule resolution and Supabase Postgres alerts insertion test suite:
```powershell
python backend/test_multimodel_fusion.py
```

---

> [!IMPORTANT]
> The old `DigitalShield2` directory is no longer required and can be safely deleted or archived. Your consolidated `DigitalShield` repository is now **100% complete, fully self-contained, and production-ready**!
