# ML-NIDS & Explainable AI: Interactive Live Demonstration Script & Defense Walkthrough

## Oral Presentation Guide, SOC Incident Triage Walkthrough & Screenshots Checklist

**Project Title:** Machine Learning Network Intrusion Detection System with Random Forest and Explainable AI (SHAP)  
**Target Audience:** Academic Evaluation Committee, Senior Cybersecurity Leadership, SOC Incident Responders  
**Estimated Walkthrough Duration:** 10 to 15 minutes  

---

## 1. Demonstration Prerequisites & Environment Setup

Before initiating the demonstration, verify that the local environment is pre-warmed:
```bash
# Terminal 1: Launch the FastAPI SOC Web Console
python run_app.py
# Opens automatically at http://localhost:8000

# Terminal 2: (Optional) Launch the Streamlit Dashboard
python -m streamlit run src/app/dashboard.py
# Opens at http://localhost:8501
```

---

## 2. Live Demonstration Script (Step-by-Step)

### Phase 1: Problem Statement & Defensive Architecture (2 Minutes)
- **Speaker Prompt:**  
  *"Good morning, esteemed committee members. Today I am presenting our Machine Learning-Based Network Intrusion Detection System with Explainable AI. Traditional rule-based intrusion detectors miss zero-day exploits, while complex neural networks act as untrustworthy black boxes that flood SOC analysts with uninterpretable false alarms. Furthermore, existing research literature frequently suffers from fatal data leakage by memorizing testbed IP addresses."*
- **Visual Action:** Open `http://localhost:8000` to the **Overview & Architecture** tab.
- **Key Talking Points:**
  - Point to the **Defensive Machine Learning Architecture** diagram: Ingestion $\to$ Sanitization $\to$ Feature Engineering $\to$ XAI Triage.
  - Highlight the KPI cards: **99.94% Holdout Test Accuracy**, **0.0425% False Positive Rate**, **0.1127% False Negative Rate**, and **5.03 microsecond latency** ($198,700\text{ flows/sec}$).
  - Emphasize the **Operational Distinction Notice**: Explain the *Base-Rate Fallacy* and why low FPR still requires human-in-the-loop explainability in production.

---

### Phase 2: Live Threat Detection Simulator (3 Minutes)
- **Speaker Prompt:**  
  *"Let us now observe the model classifying live physical traffic streams in real time."*
- **Visual Action:** Click the **Live Threat Detection** tab.
- **Demo Steps:**
  1. Click **Normal MQTT Telemetry**:
     - *Observation:* Model outputs **BENIGN TRAFFIC (NORMAL)** with $0.0\%$ attack probability. Green glowing status card. Advisory indicates normal baseline telemetry.
  2. Click **Nmap ACK Scan (-sA)**:
     - *Observation:* Model instantly flips to **CRITICAL: MALICIOUS ATTACK DETECTED** with $100.0\%$ attack probability. Red glowing alert card. Advisory recommends isolating source IP and inspecting firewall logs.
  3. Click **SPARTA Reconnaissance**:
     - *Observation:* Model flags high-volume penetration scan targeting SSH (port 22) with $100.0\%$ confidence and identifies multiclass subtype as `sparta`.

---

### Phase 3: The Explainable AI (SHAP) Investigation (4 Minutes)
- **Speaker Prompt:**  
  *"Here is the core breakthrough of our research: when an alert is flagged, our SOC analyst does not have to guess why. By clicking 'Inspect SHAP XAI Explanations', we transition from black-box prediction to cooperative game theory."*
- **Visual Action:** Click the **Explainable AI (SHAP)** tab.
- **Key Talking Points:**
  - Point to the **Local Feature Attribution Chart**: Red bars represent positive forces ($\phi_i > 0$) pushing the model toward an alert; green bars represent negative forces ($\phi_i < 0$) supporting normal traffic.
  - Explain the top drivers:
    - *`bytes_per_pkt` at 42.0:* Discloses that the single-packet probe lacks normal application payload.
    - *`bwd_min_pkt_len` at 40.0:* Exposes the 40-byte TCP RST response triggered from the closed firewall port.
    - *`bwd_num_rst_flags` = 1:* Confirms unsolicited connection reset.
  - Point to the **Automated SOC Briefing**: Read the synthesized plain-English narrative. Show how the briefing explains the mathematical prior shift ($E[f(x)] = 27.4\% \to P(\text{Attack}) = 100.0\%$).
  - Point to the **Baseline Audit Table**: Contrast the observed feature values against empirical normal averages.

---

### Phase 4: Batch Ingestion & PCAP Dissection (3 Minutes)
- **Speaker Prompt:**  
  *"In real SOC operations, traffic arrives as raw packet captures or tabular logs. Let us ingest both."*
- **Visual Action:** Return to **Live Threat Detection** and show the ingestion cards.
- **Demo Steps:**
  1. Click **Ingest Feature CSV** $\to$ Select any sample CSV.
     - *Observation:* Evaluates hundreds of flows within milliseconds, populating the **Alert Triage Queue** table with color-coded severity badges (`CRITICAL`, `HIGH`, `BENIGN`).
  2. Demonstrate Click-to-Triage:
     - In the Alert Triage Queue, click **Inspect XAI** next to any row.
     - *Observation:* The dashboard automatically navigates to Tab 4 and re-computes the exact SHAP attribution chart and briefing for that specific flow!

---

### Phase 5: Standalone Headless CLI & SIEM Forwarding (2 Minutes)
- **Speaker Prompt:**  
  *"Finally, for enterprise SIEM integration, our detector operates in a completely headless CLI mode."*
- **Visual Action:** Switch to Terminal and run:
  ```bash
  python src/models/predict.py --input data/processed/biflow_test.parquet --limit 3 --format syslog --explain
  ```
- **Key Talking Points:**
  - Point to the output **ArcSight Common Event Format (CEF)** strings:
    ```
    CEF:0|DeepMind-Cyber|ML-NIDS|1.0.0|THREAT_ALERT|Malicious Network Intrusion Detected|10|src=unknown dst=unknown spt=0 dpt=1883 proto=TCP act=ALERT prob=1.0000 cat=mqtt_bruteforce shaptop=bytes_per_pkt:0.0685|pkt_ratio:0.0563|bwd_num_bytes:0.0495
    ```
  - Highlight that each alert contains the normalized port, threat score, attack subtype, and top contributing SHAP features (`shaptop`), making it directly ingestible by Splunk or Sentinel.
  - Demonstrate cryptographic tampering prevention:
    ```bash
    python src/models/predict.py --input data/processed/biflow_test.parquet --limit 2 --verify-checksum
    ```
    Show that the model's SHA-256 is validated against `model_manifest.json` before execution, completely mitigating CWE-502 deserialization hazards.

---

## 3. Screenshots & Visual Verification Checklist

| UI Panel / State | URL / Location | Key Elements to Verify | Verification Status |
| :--- | :--- | :--- | :---: |
| **Top Global Header** | `http://localhost:8000/` | Glowing model active badge (`34 Feats`), 99.94% accuracy pill, Export SOC Report button | Verified |
| **Tab 1: Overview** | `/` (Tab 1) | 5 KPI metric cards, Defensive ML Architecture diagram, Evaluated Testbed Facts | Verified |
| **Tab 2: Dataset Intelligence** | `/` (Tab 2) | Class distribution stacked bar chart, protocol doughnut chart, flow comparison table | Verified |
| **Tab 3: Detection (Benign)** | `/` (Tab 3) | Green glowing badge (`BENIGN TRAFFIC (NORMAL)`), 0.0% probability, normal advisory | Verified |
| **Tab 3: Detection (Alert)** | `/` (Tab 3) | Red glowing badge (`CRITICAL: MALICIOUS ATTACK DETECTED`), 100.0% probability, subtype `scan_A` | Verified |
| **Tab 4: SHAP Attribution** | `/` (Tab 4) | Horizontal bar chart with red/green contributions, automated briefing narrative, baseline audit table | Verified |
| **Tab 5: Audit & Specs** | `/` (Tab 5) | Global SHAP feature importance bar chart, 2x2 confusion matrix (FP=12, FN=12) | Verified |
| **Streamlit Dashboard** | `http://localhost:8501` | Multi-tab interface, operational threshold slider ($\tau \in [0.1, 0.9]$), CSV/PCAP uploaders | Verified |
| **Swagger REST API Docs**| `http://localhost:8000/docs` | Interactive OpenAPI documentation for all 8 REST endpoints | Verified |

---

## 4. Live Demo Recovery & Troubleshooting Playbook

- **Port 8000 Conflict:** If port 8000 is occupied, launch on an alternative port:
  ```bash
  python -m uvicorn src.app.main:app --port 8080
  ```
- **Streamlit Startup Port:** If Streamlit defaults to a different port, check the console output (usually `http://localhost:8501` or `8502`).
- **Offline Fallback:** The web console has built-in offline simulation logic: if the FastAPI backend is temporarily interrupted, the client-side JavaScript continues to render interactive scenarios seamlessly.
