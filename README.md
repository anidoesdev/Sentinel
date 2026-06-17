# SENTINEL — Real-Time Multimodal Anomaly Detection

Production-grade anomaly detection system for industrial machines. Combines time-series sensor data (CMAPSS turbofan dataset) with audio signatures (MIMII pump/fan dataset) to catch equipment failures before they happen.

Built as a portfolio piece demonstrating end-to-end ML engineering: data pipelines, model training, serving, explainability, drift monitoring, streaming ingestion, and a live dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Sources                            │
│   CMAPSS (turbofan sensors)      MIMII (pump/fan audio WAVs)    │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
       ┌───────▼───────┐         ┌────────▼────────┐
       │  VAE Scorer   │         │   AST Encoder   │
       │ (time-series) │         │  + PCA + Scaler │
       │   Conv1D-VAE  │         │  (audio anomaly)│
       └───────┬───────┘         └────────┬────────┘
               │      Champion/Challenger  │
               │      Shadow Deployment    │
               └──────────┬───────────────┘
                           │
                    ┌──────▼──────┐
                    │  FastAPI    │  POST /score/{timeseries,audio}
                    │  Server     │  WS   /ws/{timeseries,replay}
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐
   │ TimescaleDB │  │    SHAP /   │  │  Evidently │
   │ Hypertables │  │  Integrated │  │    Drift   │
   │  (scores,   │  │  Gradients  │  │  Monitor   │
   │  readings)  │  │ Explainer   │  └────────────┘
   └─────────────┘  └─────────────┘
          │
   ┌──────▼──────┐     ┌───────────────────┐
   │ Redis Stream│     │  Next.js Dashboard │
   │ Ingestion   │────▶│  (live WebSocket)  │
   │  Producer / │     │  Recharts + alerts │
   │  Consumer   │     └───────────────────┘
   └─────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Conv1D-VAE for time-series | Captures temporal patterns across sensor channels; reconstruction error as anomaly score is interpretable and threshold-calibratable |
| AST (Audio Spectrogram Transformer) | Pretrained on AudioSet (527 classes); frozen feature extractor gives rich 768-dim embeddings without training on small MIMII dataset |
| PCA 768→32 + Mahalanobis distance | Dimensionality reduction before distance scoring; equivalent to squared Mahalanobis with diagonal covariance |
| Shadow deployment | Champion and challenger both score every request; only champion returned to client; challenger logged for offline A/B comparison with no production risk |
| Redis Streams over Kafka | Same guarantees (consumer groups, ACK, replay) with one fewer infrastructure dependency for a single-machine deployment |
| TimescaleDB over plain Postgres | Native time-series partitioning (hypertables), automatic chunk management, efficient range queries |
| KernelSHAP + Integrated Gradients | SHAP is model-agnostic (works on the full reconstruction-error pipeline); IG attributes at timestep×sensor resolution for VAE internals |
| Evidently KS drift detection | Reference distribution from healthy training windows; KS test per sensor; triggers alert when >30% of sensors drift |

---

## Model Results

| Modality | Dataset | Metric | Value |
|---|---|---|---|
| Audio (AST + PCA) | MIMII pump id_00 | AUROC | **0.957** |
| Audio (AST + PCA) | MIMII pump id_00 | Oracle F1 | 0.89 (threshold=61.4) |
| Time-series (VAE) | CMAPSS FD001 | AUROC | ~0.88 |
| Ensemble | CMAPSS FD001 | F1 @ threshold | See `scripts/evaluate_ensemble.py` |

Calibrated threshold (99th pct of training scores) is conservative; oracle threshold maximises F1 on the test set. AUROC is the headline metric — threshold-independent.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Models | PyTorch, HuggingFace Transformers (AST), scikit-learn (PCA) |
| Serving | FastAPI, Uvicorn, WebSockets |
| Explainability | SHAP (KernelExplainer), custom Integrated Gradients |
| Drift monitoring | Evidently |
| Streaming ingestion | Redis Streams (consumer groups, XREADGROUP/XACK) |
| Storage | TimescaleDB (PostgreSQL + hypertables) |
| Dashboard | Next.js 14, TypeScript, Tailwind CSS, Recharts |
| Experiment tracking | MLflow |
| Infrastructure | Docker Compose |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Node.js 18+ (for dashboard)

### 1. Install Python dependencies

```bash
pip install -e ".[dev]"
```

### 2. Start infrastructure

```bash
docker compose up -d
```

This starts Redis (port 6379) and TimescaleDB (port 5432). The database schema is applied automatically from `infra/init_db.sql`.

### 3. Download data

**CMAPSS (turbofan degradation):**
```
https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
```
Place files in `data/raw/`: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`

**MIMII (industrial machine audio):**
```
https://zenodo.org/record/3384388
```
Place pump data in `data/mimii/pump/` with layout: `id_00/normal/` and `id_00/abnormal/`

### 4. Train models

```bash
# Time-series VAE
python scripts/train_vae.py

# Evaluate VAE
python scripts/evaluate_vae.py

# Audio scorer (AST + PCA)
python scripts/evaluate_audio.py --machine-type pump --machine-id id_00
```

Artifacts are saved to `artifacts/vae/checkpoint.pt` and `artifacts/audio/scorer.pkl`.

### 5. Start the API server

```bash
python scripts/serve.py
# or
uvicorn sentinel.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

### 6. Start the dashboard

```bash
cd dashboard
npm install
npm run dev
```

Dashboard: http://localhost:3000

### 7. Run the streaming pipeline

In separate terminals:

```bash
# Consumer: reads from Redis, scores, writes to TimescaleDB
python scripts/stream_consumer.py --unit-ids 1

# Producer: publishes synthetic sensor degradation sequence
python scripts/stream_producer.py --mode synthetic --unit-id 1 --steps 150
```

---

## API Reference

### `GET /health`
Returns loaded models and server status.

### `POST /score/timeseries`
Score a window of sensor readings.

```json
{
  "unit_id": 1,
  "readings": [[641.82, 0.023, ...], ...]
}
```

Returns: `{"unit_id": 1, "anomaly_score": 0.0042, "is_anomalous": false, "threshold": 0.05, "modality": "timeseries"}`

### `POST /score/audio`
Score a WAV file (multipart upload). Accepts 16kHz mono/stereo WAV.

### `WS /ws/timeseries/{unit_id}`
Streaming scoring. Send `{"sensors": [v1, ..., v14]}` repeatedly; receive scores once buffer fills.

### `WS /ws/replay`
Synthetic degradation demo. Streams 150 steps (healthy → degrading → anomalous) for dashboard demos without live sensor data. Query params: `speed_ms` (default 400), `total_steps` (default 150).

---

## Shadow Deployment

Place a challenger checkpoint at `artifacts/vae/challenger.pt`. On every `/score/timeseries` request, both champion and challenger will score the input. The champion result is returned to the client; the challenger result is written to TimescaleDB with `is_shadow=TRUE`.

Query challenger performance:

```sql
SELECT
    model_version,
    AVG(score)              AS mean_score,
    SUM(is_anomalous::int)  AS n_anomalies,
    COUNT(*)                AS total
FROM anomaly_scores
WHERE time > NOW() - INTERVAL '1 hour'
GROUP BY model_version, is_shadow;
```

---

## Repository Structure

```
sentinel/
├── data/           # Data loading (CMAPSS, MIMII auto-detect layouts)
├── models/         # VAE architecture (Conv1D encoder/decoder)
├── inference/      # VAEAnomalyScorer, ASTEncoder + AudioAnomalyScorer
├── serving/        # FastAPI app, model store, Pydantic schemas
├── explanations/   # KernelSHAP, Integrated Gradients
├── monitoring/     # Evidently drift monitor
├── streaming/      # Redis Streams producer + consumer
├── storage/        # TimescaleDB writer
scripts/
├── train_vae.py
├── evaluate_vae.py, evaluate_audio.py, evaluate_ensemble.py
├── serve.py
├── explain.py
├── monitor_drift.py
├── stream_producer.py
├── stream_consumer.py
dashboard/          # Next.js 14 live anomaly dashboard
infra/
├── init_db.sql     # TimescaleDB hypertable schema
docker-compose.yml  # Redis + TimescaleDB
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT
