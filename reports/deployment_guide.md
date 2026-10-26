# ML-NIDS Production Deployment Guide

A step-by-step production operations manual for deploying the **Machine Learning-Based Network Intrusion Detection System (ML-NIDS)** across Docker, Cloud VMs (AWS/GCP/Azure), on-premises Linux servers, and live network TAP/SPAN ports.

---

## Deployment Architecture Overview

```
                        [ Ingress Traffic / Internet / WAN ]
                                       |
                   +-------------------v-------------------+
                   |         Nginx Reverse Proxy          |
                   |   (SSL/TLS Termination, Rate Limiting)|
                   +---------+-------------------+---------+
                             |                   |
               Port 443 -> 8000                  Port 443 -> 8501
                             |                   |
            +----------------v-----+    +--------v----------------+
            |      FastAPI REST    |    |   Streamlit SOC Console |
            |      Engine / Web    |    |   Interactive Analytics |
            +----------------+-----+    +--------+----------------+
                             |                   |
                             +---------+---------+
                                       |
                   +-------------------v-------------------+
                   |     ML Inference & SHAP Engine        |
                   |  - 150-Tree Random Forest Classifier  |
                   |  - SHAP TreeExplainer Local Engine    |
                   |  - SHA-256 Validated Model Bundle     |
                   +-------------------+-------------------+
                                       |
                   +-------------------v-------------------+
                   |         SIEM / Syslog Forwarder       |
                   |   (ArcSight CEF / RFC 5424 Syslog)    |
                   +---------------------------------------+
                                       |
                   [ Splunk / Elastic / Sentinel / QRadar ]
```

---

## Option 1: Turnkey Containerized Deployment (Docker & Compose)

**Best for:** Rapid deployment on Linux, Windows Server, or macOS with zero dependency friction.

### Prerequisites
- Docker Engine $\ge 24.0$
- Docker Compose $\ge 2.20$

### 1. Build and Start the Containers
From the root of the project directory:
```bash
# Build the images and start the services in detached mode
docker-compose up -d --build
```

### 2. Verify Container Health
```bash
# Check service status and health checks
docker-compose ps

# Output:
# NAME                IMAGE          COMMAND                  SERVICE          STATUS
# ml-nids-api         aismini_api    "uvicorn src.app.m..."   nids-api         Up (healthy)
# ml-nids-dashboard   aismini_dash   "python -m streaml..."   nids-dashboard   Up
```

### 3. Access the Consoles
- **FastAPI Web Console & REST API:** `http://<SERVER_IP>:8000`
- **Interactive Swagger Documentation:** `http://<SERVER_IP>:8000/docs`
- **Streamlit SOC Analyst Dashboard:** `http://<SERVER_IP>:8501`

### 4. Container Management Commands
```bash
# View unified real-time logs
docker-compose logs -f

# View API logs only
docker-compose logs -f nids-api

# Stop services
docker-compose down

# Restart services
docker-compose restart
```

---

## Option 2: Linux Cloud Server Deployment (Ubuntu / Debian / AWS EC2 / GCP)

**Best for:** Production cloud virtual machines, dedicated servers, or on-premises enterprise Linux.

### 1. System Preparation
```bash
# Update package repositories
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.11/3.12, pip, venv, libpcap, and Nginx
sudo apt-get install -y python3 python3-pip python3-venv libpcap-dev gcc nginx git
```

### 2. Clone Repository & Setup Virtual Environment
```bash
# Create application directory
sudo mkdir -p /opt/ml-nids
sudo chown -R $USER:$USER /opt/ml-nids
cd /opt/ml-nids

# Clone the repository (or copy files via rsync/scp)
git clone <YOUR_REPO_URL> .

# Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install all locked dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Model Integrity & Run Tests
```bash
# Verify that all 48 test suites pass on the server
python -m pytest
```

---

### 4. Configure Systemd Services (Automatic Startup & Restart)

#### Service 1: FastAPI REST Engine (`/etc/systemd/system/nids-api.service`)
```ini
[Unit]
Description=ML-NIDS FastAPI Inference Engine
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/ml-nids
Environment="PATH=/opt/ml-nids/venv/bin"
Environment="PYTHONPATH=/opt/ml-nids"
ExecStart=/opt/ml-nids/venv/bin/uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

#### Service 2: Streamlit Dashboard (`/etc/systemd/system/nids-dashboard.service`)
```ini
[Unit]
Description=ML-NIDS Streamlit SOC Dashboard
After=network.target nids-api.service

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/ml-nids
Environment="PATH=/opt/ml-nids/venv/bin"
Environment="PYTHONPATH=/opt/ml-nids"
ExecStart=/opt/ml-nids/venv/bin/python -m streamlit run src/app/dashboard.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

#### Enable and Start the Services:
```bash
# Set ownership
sudo chown -R www-data:www-data /opt/ml-nids

# Reload systemd and start services
sudo systemctl daemon-reload
sudo systemctl enable --now nids-api.service
sudo systemctl enable --now nids-dashboard.service

# Check status
sudo systemctl status nids-api.service
sudo systemctl status nids-dashboard.service
```

---

### 5. Nginx Reverse Proxy & SSL/TLS Setup

Create `/etc/nginx/sites-available/ml-nids`:
```nginx
# Upstream definitions
upstream api_backend {
    server 127.0.0.1:8000;
}

upstream dashboard_backend {
    server 127.0.0.1:8501;
}

server {
    listen 80;
    server_name nids.yourdomain.com;

    # Redirect HTTP to HTTPS (uncomment when SSL is installed)
    # return 301 https://$host$request_uri;

    # FastAPI REST Engine & Web Console
    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Streamlit SOC Dashboard
    location /soc/ {
        proxy_pass http://dashboard_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for Streamlit live updates
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

Enable site and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/ml-nids /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### Optional: Enable Free SSL with Let's Encrypt Certbot
```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d nids.yourdomain.com
```

---

## Option 3: Headless SIEM Pipeline Deployment (ArcSight CEF / Syslog)

**Best for:** Security Operations Centers feeding automated alerts directly into Splunk, Microsoft Sentinel, IBM QRadar, or Elastic SIEM.

### 1. Ingest Batch Flows and Forward to SIEM
Run headless inference against raw biflow logs or extracted sessions, generating CEF syslog records:
```bash
python src/models/predict.py \
  --input /path/to/incoming_flows.parquet \
  --format cef \
  --threshold 0.50 \
  --output /var/log/nids_cef.log
```

### 2. Forward to Remote SIEM Syslog Receiver
Configure `rsyslog` (`/etc/rsyslog.d/50-ml-nids.conf`):
```text
# Monitor the ML-NIDS CEF log file and forward to central SIEM
module(load="imfile")

input(type="imfile"
      File="/var/log/nids_cef.log"
      Tag="ml-nids:"
      Severity="notice"
      Facility="local0")

# Forward to SIEM collector over TCP port 514
local0.* @@siem-collector.internal.corp:514
```
Restart rsyslog:
```bash
sudo systemctl restart rsyslog
```

---

## Option 4: Live Network TAP / SPAN Port Sniffing Deployment

**Best for:** Connecting an on-premises physical appliance or hypervisor mirror port to analyze line traffic in real time.

```
       [ Core Switch / Router ]
                 |
      [ Mirror / SPAN Port ]
                 |
         eth1 (Promiscuous)
                 |
   +-------------v-------------+
   |   PcapFlowExtractor       |  (Streaming packet reassembly)
   +-------------+-------------+
                 |
   +-------------v-------------+
   |  NIDSPredictionPipeline   |  (Leakage-safe 34-feature extraction)
   +-------------+-------------+
                 |
   +-------------v-------------+
   |   Ensemble + SHAP Engine  |  (5.03 µs / flow classification)
   +---------------------------+
```

### 1. Configure the Network Interface in Promiscuous Mode
```bash
# Enable promiscuous mode on sniffing interface (e.g. eth1)
sudo ip link set eth1 promisc on
sudo ip link set eth1 up
```

### 2. Execute Real-Time Streaming Sniffer Daemon
Execute the headless extractor against a rotating ring buffer of PCAPs generated by `tcpdump`:
```bash
# Terminal 1: Capture traffic ring buffer (1-minute slices)
sudo tcpdump -i eth1 -n -s 0 -G 60 -W 5 -w '/var/nids/live_%%Y%%m%%d%%H%%M%%S.pcap'

# Terminal 2: Headless Python watch loop
python src/models/predict.py \
  --input /var/nids/ \
  --format cef \
  --output /var/log/nids_alerts.log
```

---

## Production Security Hardening Checklist

| Hardening Item | Implementation Detail | Status |
| :--- | :--- | :---: |
| **Model Deserialization (CWE-502)** | Cryptographic SHA-256 verification of `nids_bundle.joblib` before unpickling | **Enforced** |
| **API Attack Surface** | Non-root container execution; CORS origin restriction | **Enforced** |
| **Zero Identifiers Memorized** | IP addresses (`ip_src`, `ip_dst`) purged; cannot leak internal network topologies | **Enforced** |
| **Fail-Safe Thresholds** | Configurable $\tau \in [0.05, 0.95]$ depending on enterprise risk posture | **Enforced** |
| **TLS/SSL Encryption** | Mandatory HTTPS termination via Nginx for browser traffic | **Documented** |
| **Rate Limiting** | Nginx `limit_req_zone` (e.g., 20 requests/sec per IP) | **Documented** |
