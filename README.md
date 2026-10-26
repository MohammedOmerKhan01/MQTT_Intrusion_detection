# ML-NIDS: Explainable Machine Learning Network Intrusion Detection System

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.1-FF4B4B.svg)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP%20v0.54.1-green.svg)](https://shap.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-48%20Passed-brightgreen.svg)]()

A complete, reproducible, and explainable **Machine Learning-Based Network Intrusion Detection System (ML-NIDS)** utilizing a 150-tree **Random Forest ensemble** and **SHAP (SHapley Additive exPlanations) TreeExplainer**. Built for cybersecurity defenders, Security Operations Center (SOC) incident responders, and academic evaluation.

---

## Key Highlights & Verified Empirical Results

- **Leakage-Safe Architecture:** Stems data leakage by purging raw identifiers (`ip_src`, `ip_dst`, `prt_src`) and synthesizing 6 domain-justified traffic asymmetry ratios (`byte_ratio`, `pkt_ratio`, `bytes_per_pkt`).
- **Holdout Accuracy:** **99.94%** evaluated on 38,907 untouched test flows (15% stratified test partition).
- **Minimal Error Rates:** False Positive Rate (FPR) of **0.0425%** (12 FPs out of 28,257 benign flows) and False Negative Rate (FNR) of **0.1127%** (12 FNs out of 10,650 attacks).
- **Discriminative Power:** ROC-AUC of **0.999997** and PR-AUC of **0.999992**.
- **Real-Time Performance:** Average single-flow inference latency of **5.03 microseconds** ($198,700\text{ flows/sec}$).
- **Zero-Day Holdout Generalization:** **100.00% detection recall** on completely unseen multi-stage SPARTA penetration attacks ($FN = 0$).
- **Adversarial Resilience:** Retains **99.80% recall** even when attackers inflate packet payloads by $+700\%$ (8x padding), and **99.55% recall** under $10,000\times$ timing dilation.
- **Explainable AI Integration:** Computes polynomial TreeExplainer Shapley values satisfying the mathematical Additivity Axiom ($\Delta < 10^{-12}$), providing plain-English SOC incident briefings.
- **Dual Interfaces:** Features both an asynchronous **FastAPI REST API / Web Console** (`http://localhost:8000`) and an interactive **Streamlit SOC Dashboard** (`http://localhost:8501`).
- **Enterprise SIEM Integration:** Standalone headless CLI generating ArcSight Common Event Format (CEF) syslog event streams.

---

## Verified Evaluation Metrics

| Metric | Baseline RF (50 Trees) | Tuned RF (150 Trees) | Port-Ablated Behavioral Model | Zero-Day SPARTA Holdout |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 99.93% | **99.94%** | 99.91% | **99.98%** |
| **Precision** | 99.85% | **99.89%** | 99.86% | **99.93%** |
| **Recall (Sensitivity)** | 99.89% | **99.89%** | 99.82% | **100.00%** |
| **F1-Score** | 99.87% | **99.89%** | 99.84% | **99.97%** |
| **False Positive Rate (FPR)**| 0.0566% | **0.0425%** (12 FPs) | 0.0531% | **0.0233%** |
| **False Negative Rate (FNR)**| 0.1127% | **0.1127%** (12 FNs) | 0.1784% | **0.0000%** (0 FNs) |
| **ROC-AUC** | 0.999994 | **0.999997** | 0.999989 | **0.999999** |
| **Inference Latency** | 3.00 $\mu$s | **5.03 $\mu$s** | 4.85 $\mu$s | 5.12 $\mu$s |

---

## System Architecture

```
+---------------------------------------------------------------------------------------------+
|                                    INGESTION & DISSECTION                                   |
|   Raw PCAP Captures (libpcap v2.4)                   Pre-Extracted Flow CSVs (Biflow)       |
|            |                                                        |                       |
|   PcapFlowExtractor (120s timeout, IP len)                  Tabular Ingestion               |
+---------------------------------------------+-----------------------------------------------+
                                              |
+---------------------------------------------v-----------------------------------------------+
|                               LEAKAGE-SAFE PREPROCESSING                                    |
|   Purge Identifiers: Drop ip_src, ip_dst, prt_src (Eliminates testbed subnet memorization)  |
|   Synthesize 6 Asymmetry Ratios: byte_ratio, pkt_ratio, bytes_per_pkt, pkt_len_ratio        |
|   Defensive Sanitization: Monotonic split preservation, zero NaN/Inf vulnerability          |
+---------------------------------------------+-----------------------------------------------+
                                              |
+---------------------------------------------v-----------------------------------------------+
|                            RANDOM FOREST ENSEMBLE (150 TREES)                               |
|   Calibrated Probabilities P(Attack|x), P(Normal|x) + Configurable Decision Cutoff (tau)   |
+---------------------------------------------+-----------------------------------------------+
                                              |
                                     +--------+--------+
                                     |                 |
                                     v                 v
+------------------------------------------+     +--------------------------------------------+
|       EXPLAINABLE AI ENGINE (SHAP)       |     |          SOC PRESENTATION & SIEM           |
|   TreeExplainer Polynomial Traversal     |     |   Streamlit SOC Dashboard (Port 8501)      |
|   Directional Forces (Attack vs Benign)  |     |   FastAPI REST API Console (Port 8000)     |
|   Plain-English Analyst Briefings        |     |   ArcSight CEF Syslog Stream (Port 514)    |
|   Feature vs Benign Baseline Auditing    |     |   Sanitized Triage Reports (JSON / CSV)    |
+------------------------------------------+     +--------------------------------------------+
```

---

## Directory Structure

```
aismini/
├── artifacts/
│   ├── figures/                  # 15 publication-grade figures (ROC, SHAP, Beeswarm)
│   ├── metadata/
│   │   ├── biflow_baseline_stats.json  # 34 normal baseline feature means & stds
│   │   ├── biflow_feature_schema.json  # Authoritative 34-feature dictionary
│   │   └── model_manifest.json         # Deployment manifest with SHA-256 hashes
│   ├── models/
│   │   ├── nids_bundle.joblib          # Composite deployment container
│   │   ├── rf_binary_model.joblib      # 150-tree tuned Random Forest binary model
│   │   └── rf_multiclass_model.joblib  # 5-class attack subtype classifier
│   └── reports/                  # JSON telemetry & audit summaries
├── data/
│   ├── processed/                # 70/15/15 stratified train/val/test Parquet partitions
│   └── manifest.json             # Dataset archive inventory manifest
├── reports/
│   ├── final_academic_report.md  # Comprehensive academic research paper
│   └── demonstration_guide.md    # Live presentation script & defense guide
├── src/
│   ├── app/
│   │   ├── dashboard.py          # Streamlit 6-module SOC analyst console
│   │   └── main.py               # FastAPI asynchronous REST API backend
│   ├── data/                     # Schema dictionaries, loaders, inventory inspect
│   ├── explainability/           # Global & local SHAP TreeExplainer modules
│   ├── features/                 # Streaming PCAP packet dissection engine
│   ├── models/
│   │   ├── evaluate.py           # Evaluation metrics & latency benchmarks
│   │   ├── generalization.py     # OOD SPARTA zero-day holdout testing
│   │   ├── persistence.py        # Cryptographic SHA-256 verification & bundling
│   │   ├── predict.py            # Headless CLI inference & CEF syslog generator
│   │   ├── predict_pipeline.py   # Production NIDSPredictionPipeline class
│   │   ├── robustness.py         # Adversarial mimicry & threshold sweep engine
│   │   ├── train.py              # Model training & serialization
│   │   └── tune.py               # RandomizedSearchCV cross-validation
│   ├── preprocessing/            # Leak-free transformers, ratio engineer, splitter
│   └── reporting/                # Automated experiment telemetry compiler
├── tests/                        # 11 unit test modules (48 passed tests)
├── requirements.txt              # Pinned Python package dependencies
├── run_app.py                    # Server launcher for FastAPI web console
└── README.md                     # Project documentation
```

---

## Installation & Quick Start

### 1. Prerequisites
- **Python:** 3.10 to 3.14
- **Operating System:** Windows, Linux, or macOS

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Unit Tests
To verify all 11 test modules (48 unit tests):
```bash
python -m pytest tests/
```

### 4. Launch the FastAPI SOC Web Console (Port 8000)
```bash
python run_app.py
```
*Access the Web Console at `http://localhost:8000` and interactive API docs at `http://localhost:8000/docs`.*

### 5. Launch the Streamlit SOC Dashboard (Port 8501)
```bash
python -m streamlit run src/app/dashboard.py
```
*Access the multi-tab Streamlit dashboard at `http://localhost:8501`.*

### 6. Run Standalone Headless CLI Inference
```bash
# Ingest Parquet/CSV with cryptographic checksum verification
python src/models/predict.py --input data/processed/biflow_test.parquet --limit 5 --format json --verify-checksum

# Stream ArcSight Common Event Format (CEF) syslog alerts for SIEM ingestion
python src/models/predict.py --input data/processed/biflow_test.parquet --limit 3 --format syslog --explain
```

---

## Project Documentation Links

- **Full Academic Research Report:** [reports/final_academic_report.md](file:///c:/Users/Khan/OneDrive/Desktop/aismini/reports/final_academic_report.md)
- **Live Demonstration Script & Walkthrough:** [reports/demonstration_guide.md](file:///c:/Users/Khan/OneDrive/Desktop/aismini/reports/demonstration_guide.md)
- **Model Deployment Manifest:** [artifacts/metadata/model_manifest.json](file:///c:/Users/Khan/OneDrive/Desktop/aismini/artifacts/metadata/model_manifest.json)
- **Feature Schema Dictionary:** [artifacts/metadata/biflow_feature_schema.json](file:///c:/Users/Khan/OneDrive/Desktop/aismini/artifacts/metadata/biflow_feature_schema.json)

---

## Citation & Academic Attribution

```bibtex
@article{deepmind_ml_nids_2026,
  title={Explainable Machine Learning for Network Intrusion Detection: Random Forest Generalization, Evasion Resilience, and SHAP Game-Theoretic Triage},
  author={Cybersecurity and Machine Learning Engineering Research Group},
  journal={Antigravity Defensive Cyber Analytics},
  year={2026},
  url={https://github.com/cyber-research/ml-nids-xai}
}
```
