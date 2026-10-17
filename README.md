# Federated ScamShield

> Privacy-preserving cross-bank fraud detection powered by Federated Learning on IBM LinuxONE

---

## Problem Statement

Financial fraud costs the global economy over $5 trillion annually. Traditional fraud detection systems operate in silos -- each bank trains its own model on its own data, missing cross-institutional fraud patterns like mule ring networks that span multiple banks. Centralising data is not an option: regulatory frameworks (GDPR, PSD2, India's DPDP Act) prohibit pooling sensitive transaction records.

**Federated ScamShield** solves this by enabling collaborative model training across banks without any raw data leaving the institution. The result: a globally informed fraud detector with mathematically guaranteed privacy.

---

## Solution Architecture

```
                          ┌─────────────────────────────────┐
                          │      IBM LinuxONE Server        │
                          │   (s390x / Secure Execution)    │
                          │                                 │
                          │  ┌───────────────────────────┐  │
                          │  │   Aggregation Server      │  │
                          │  │   (FedAvg Coordinator)    │  │
                          │  │                           │  │
                          │  │  ┌─────────────────────┐  │  │
                          │  │  │   Global Model       │  │  │
                          │  │  │   (ScamMLP 23-dim)   │  │  │
                          │  │  └─────────────────────┘  │  │
                          │  │                           │  │
                          │  │  ┌─────────────────────┐  │  │
                          │  │  │  HMAC Watchlist      │  │  │
                          │  │  │  (Mule Ring Detect)  │  │  │
                          │  │  └─────────────────────┘  │  │
                          │  └───────┬───┬───┬───┬───────┘  │
                          │          │   │   │   │          │
                          └──────────┼───┼───┼───┼──────────┘
                 Encrypted Gradients │   │   │   │  (No raw data)
                    ┌────────────────┘   │   │   └────────────────┐
                    │          ┌─────────┘   └─────────┐          │
               ┌────▼────┐ ┌──▼──────┐ ┌──────▼──┐ ┌───▼─────┐
               │ BANK A  │ │ BANK B  │ │ BANK C  │ │ BANK D  │
               │         │ │         │ │         │ │         │
               │ Local   │ │ Local   │ │ Local   │ │ Local   │
               │ Training│ │ Training│ │ Training│ │ Training│
               │         │ │         │ │         │ │         │
               │ 1.09M   │ │ 1.09M   │ │ 1.09M   │ │ 1.09M   │
               │ records  │ │ records  │ │ records  │ │ records  │
               └─────────┘ └─────────┘ └─────────┘ └─────────┘
                    ▲            ▲            ▲            ▲
                    │            │            │            │
              Private Data  Private Data  Private Data  Private Data
              (Never Shared)
```

---

## Experimental Design

Five setups were evaluated to rigorously measure the value of federation, watchlist sharing, and differential privacy:

```
Setup A ─── Local Baseline ──────────────────── Each bank trains alone
   │
Setup B ─── Federated Learning ──────────────── FedAvg across 4 banks
   │                                              (no raw data shared)
Setup C ─── Centralised ─────────────────────── All data pooled
   │                                              (privacy-violating upper bound)
Setup D ─── Federated + Watchlist ───────────── Setup B + HMAC-hashed
   │                                              cross-bank mule ring detection
Setup E ─── Federated + Differential Privacy ── Setup D + per-sample gradient
                                                  clipping + Gaussian noise
                                                  (formal privacy guarantee)
```

---

## Results

### Performance Comparison

| Setup | Description | PR-AUC | ROC-AUC | F1 Score |
|:-----:|-------------|:------:|:-------:|:--------:|
| A | Local Baseline (avg across banks) | 0.9750 | 0.9827 | -- |
| B | Federated Learning | 0.9789 | 0.9891 | 0.9831 |
| C | Centralised (upper bound) | 0.9849 | 0.9942 | 0.9864 |
| D | Federated + Watchlist | 0.9789 | -- | -- |
| E | Federated + DP | **0.9914** | **0.9983** | -- |

### Key Findings

- **Federation closes the gap**: Setup B reaches 99.4% of centralised PR-AUC without sharing any raw data
- **Watchlist catches mule rings**: Setup D achieves 100% recall on known mule ring networks (20/20 rings detected)
- **Privacy at no cost**: Setup E adds formal differential privacy (epsilon = 14.17, delta = 1e-5) while actually improving PR-AUC to 0.9914
- **Convergence in 5 rounds**: The federated model stabilises within 5 communication rounds, each taking ~23 seconds

### Mule Ring Detection (Setup D)

| Metric | Value |
|--------|-------|
| Rings Detected | 20 / 20 |
| Ring Recall | 100% |
| Detection Method | HMAC-hashed account matching |

### Differential Privacy Budget (Setup E)

| Parameter | Value |
|-----------|-------|
| Epsilon | 14.17 |
| Delta | 1e-5 |
| Noise Mechanism | Gaussian |
| Gradient Clipping | Per-sample, max norm = 1.0 |

---

## Model Architecture

**ScamMLP** -- a multi-layer perceptron optimised for tabular fraud detection:

```
Input (23 features)
    │
    ▼
┌──────────────────┐
│  Linear(23, 128) │
│  BatchNorm1d     │
│  ReLU            │
│  Dropout(0.3)    │
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│  Linear(128, 64) │
│  BatchNorm1d     │
│  ReLU            │
│  Dropout(0.3)    │
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│  Linear(64, 1)   │
│  Sigmoid         │
└────────┬─────────┘
         │
    ▼
Output: Fraud Probability [0, 1]
```

### Feature Engineering (23 dimensions)

| Index | Feature | Description |
|:-----:|---------|-------------|
| 0 | log_amount | Log-transformed transaction amount |
| 1 | hour_norm | Hour of day normalised to [0, 1] |
| 2 | is_international | Cross-border transaction flag |
| 3-6 | channel_* | One-hot: online, ATM, branch, mobile |
| 7 | night_flag | Transaction between 10 PM and 5 AM |
| 8 | intl_amount | Interaction: log_amount x is_international |
| 9 | high_value | Amount exceeds 10,000 threshold |
| 10-22 | derived_* | Velocity, frequency, and statistical features |

---

## Federated Learning Protocol

```
Round r:
  1. Server broadcasts global model weights W(r) to all banks
  2. Each bank k:
     a. Initialises local model with W(r)
     b. Trains on private data for E local epochs
     c. (If DP enabled) Clips per-sample gradients, adds Gaussian noise
     d. Computes update: delta_k = W_local - W(r)
     e. Sends encrypted delta_k to server
  3. Server aggregates: W(r+1) = W(r) + (1/K) * sum(delta_k)
  4. Repeat for R rounds
```

**Privacy guarantees at each step:**
- Step 2e: Only model gradients are transmitted, never raw data
- Step 2c: Differential privacy provides formal (epsilon, delta) guarantee
- Step 2e: Gradients encrypted in transit
- Watchlist: Account IDs are HMAC-hashed before cross-bank comparison

---

## System Design

### Deployment on IBM LinuxONE

```
┌──────────────────────────────────────────────────┐
│               IBM LinuxONE Server                │
│          Ubuntu 24.04 LTS (s390x)                │
│          3 cores / 8 GB RAM                      │
│                                                  │
│   ┌──────────────┐     ┌───────────────────┐     │
│   │  FastAPI      │     │  Streamlit        │     │
│   │  (port 8000)  │◄───►│  Dashboard        │     │
│   │               │     │  (port 8501)      │     │
│   │  /predict     │     │                   │     │
│   │  /predict/    │     │  - KPI Strip      │     │
│   │    batch      │     │  - Convergence    │     │
│   │  /health      │     │  - Confusion Mat  │     │
│   │               │     │  - DP Privacy     │     │
│   │  ScamMLP      │     │  - Prediction     │     │
│   │  (model.pt)   │     │    Demo           │     │
│   └──────────────┘     │  - Batch Screen   │     │
│                         └───────────────────┘     │
│                                                   │
│   ┌──────────────────────────────────────────┐    │
│   │  Training Pipeline (train.py)            │    │
│   │  Setups A → B → C → D → E               │    │
│   │  Results → results/metrics.json          │    │
│   └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

### API Endpoints

| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| GET | `/health` | -- | `{"status": "ok", "model_loaded": true, "n_features": 23}` |
| POST | `/predict` | `{"features": [23 floats]}` | `{"fraud_probability": float, "is_fraud": bool, "threshold": 0.5}` |
| POST | `/predict/batch` | `{"batch": [{"features": [...]}, ...]}` | List of predictions |

---

## Dashboard

The Streamlit dashboard provides real-time monitoring and an interactive prediction demo:

- **KPI Strip** -- PR-AUC across all 5 experimental setups at a glance
- **Live Transaction Flow** -- Animated visualisation of cross-bank transaction screening
- **Performance Comparison** -- Side-by-side bar charts and confusion matrices for all setups
- **Convergence Tracking** -- PR-AUC per federated round and per-bank loss curves
- **Differential Privacy** -- Privacy budget visualisation with epsilon/delta display
- **Pipeline Timings** -- Training duration breakdown on IBM LinuxONE hardware
- **Prediction Demo** -- Interactive fraud scoring with preset scenarios (legitimate, suspicious, high-risk)
- **Batch Screening** -- Simultaneous analysis of multiple transactions via API

---

## Project Structure

```
scamshield/
├── api.py                  # FastAPI prediction server
├── app.py                  # Streamlit dashboard (dark theme)
├── train.py                # Training pipeline (all 5 setups)
├── model.py                # ScamMLP model definition
├── data_utils.py           # Data loading and feature engineering
├── fed_utils.py            # Federated learning utilities (FedAvg)
├── watchlist.py            # HMAC-based watchlist module
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Dark theme configuration
├── results/
│   ├── metrics.json        # Experiment results (all 5 setups)
│   └── model.pt            # Trained model weights
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch (CPU build for s390x)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/federated-scamshield.git
cd federated-scamshield
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Training

```bash
python train.py
```

Executes all 5 experimental setups sequentially and writes results to `results/metrics.json`.

### Start API Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Start Dashboard

```bash
streamlit run app.py --server.port 8501
```

### Example API Call

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "features": [
        9.21, 0.125, 1.0,       # log_amount, hour_norm, is_international
        0, 0, 0, 1,              # channel: mobile
        1, 9.21, 1,              # night_flag, intl_amount, high_value
        0, 0, 0, 0, 0, 0, 0,    # derived features
        0, 0, 0, 0, 0, 0
    ]
})

print(response.json())
# {
#   "fraud_probability": 0.87,
#   "is_fraud": true,
#   "threshold": 0.5
# }
```

---

## Technical Highlights

### Why IBM LinuxONE?

- **Hardware security**: Secure Execution environment for confidential computing
- **Enterprise reliability**: Designed for 99.999% uptime financial workloads
- **Cryptographic acceleration**: Hardware-accelerated encryption for gradient protection
- **Scalability**: Vertical scaling for aggregation-heavy federated workloads

### Why Federated Learning over Centralised?

| Aspect | Centralised | Federated (Ours) |
|--------|-------------|-------------------|
| Data sharing | All data pooled | No data leaves the bank |
| Regulatory compliance | Violates GDPR/DPDP | Fully compliant |
| Single point of failure | Yes | No |
| Mule ring detection | Possible | Possible (via HMAC watchlist) |
| PR-AUC achieved | 0.9849 | 0.9914 (with DP) |

### Privacy Stack

```
Layer 1: Data Isolation ────── Raw data never leaves the bank
Layer 2: Gradient Encryption ─ Updates encrypted in transit
Layer 3: Differential Privacy ─ Gaussian noise + gradient clipping
Layer 4: HMAC Watchlist ────── Account IDs hashed before sharing
Layer 5: Secure Execution ──── IBM LinuxONE hardware enclave
```

---

## Training Performance

All benchmarks on IBM LinuxONE Community Cloud (3 CPU cores, 8 GB RAM, s390x):

| Phase | Duration |
|-------|----------|
| Data preparation | ~5s |
| Setup A (Local) | ~45s |
| Setup B (Federated) | ~115s |
| Setup C (Centralised) | ~30s |
| Setup D (Fed + Watchlist) | ~120s |
| Setup E (Fed + DP) | ~140s |
| Total pipeline | ~8 min |

---

## References

- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg, 2017)
- Abadi et al., "Deep Learning with Differential Privacy" (2016)
- IBM LinuxONE Secure Execution documentation

---

## License

MIT

## Author

**Yaswanth K B** -- IBM Z Datathon 2026 (AI Secured Track)
