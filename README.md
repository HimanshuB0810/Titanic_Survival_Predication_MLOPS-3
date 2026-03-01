
---

# Titanic Survival Prediction MLOps Project

This project implements an end-to-end MLOps pipeline for predicting passenger survival on the Titanic. It integrates automated data ingestion, orchestration with Airflow, drift detection with Alibi, and a comprehensive monitoring stack using Prometheus and Grafana.

## 🚀 Features

* **Automated Pipeline**: End-to-end training pipeline from data ingestion to model training.
* **Orchestration**: Managed workflows using Apache Airflow and Astronomer for data extraction and processing.
* **Storage & Feature Store**: Integration with MinIO for S3-compatible object storage and Redis for low-latency feature serving.
* **Web Interface**: A Flask-based web application for user-friendly survival predictions.
* **Advanced Monitoring**: Integrated Prometheus for metric collection and **Grafana** for real-time visualization.
* **Drift Detection**: Implements Kolmogorov-Smirnov (KS) drift detection using the `alibi-detect` library.

## 🏗️ Project Structure

```text
├── artifacts/              # Data and trained model binaries (.pkl)
├── config/                 # Configuration for paths and databases
├── dags/                   # Airflow DAGs for data extraction and orchestration
├── notebook/               # Jupyter notebooks for EDA and experimentation
├── pipeline/               # Training pipeline scripts
├── src/                    # Core source code (Ingestion, Processing, Training)
├── static/ & templates/    # Frontend files for the Flask application
├── application.py          # Flask entry point with drift detection
├── docker-compose.yml      # Docker multi-container setup (Monitoring, Storage)
└── prometheus.yml          # Prometheus configuration

```

## 🛠️ Tech Stack

* **Language**: Python 3.x
* **ML & Analytics**: Scikit-Learn, Alibi-detect
* **Web Framework**: Flask
* **Orchestration**: Apache Airflow / Astronomer
* **Data & Storage**: MinIO, Redis
* **Monitoring & Visualization**: Prometheus, **Grafana**
* **DevOps**: Docker & Docker Compose

## 🚦 Getting Started

### Prerequisites

* Docker and Docker Compose
* Astronomer CLI (optional, for Airflow management)

### Installation & Setup

1. **Clone the Repository**:
```bash
git clone <repository-url>
cd titanic_survival_predication_mlops-3

```


2. **Start Services with Docker**:
Launch the entire stack (Airflow, MinIO, Prometheus, Grafana, and the Flask app):
```bash
docker-compose up -d

```


3. **Access the Components**:
* **Web App**: `http://localhost:5000`
* **Grafana Dashboard**: `http://localhost:3000`
* **Prometheus**: `http://localhost:9090`
* **Airflow UI**: `http://localhost:8080`
* **MinIO**: `http://localhost:9000`



## 📊 Monitoring & Visualization

* **Prometheus**: Scrapes system metrics and model performance data as defined in `prometheus.yml`.
* **Grafana**: Connects to Prometheus as a data source to provide visual dashboards for monitoring model health, resource usage, and drift detection alerts.

## 🔄 Pipeline Workflow

1. **Data Ingestion**: Data is pulled from source and stored in MinIO.
2. **Processing**: Raw data is cleaned and features are stored in the Redis Feature Store.
3. **Training**: A Random Forest model is trained and saved to `artifacts/models/`.
4. **Deployment**: The Flask app loads the model and monitors for data drift using `KSDrift`.