# Workflow-CI untuk Training & Deployment Model MLflow

Repositori ini berisi workflow CI/CD otomatis untuk melakukan training model Machine Learning menggunakan MLflow Project. Workflow ini tidak hanya melakukan training, tetapi juga menyimpan artefak model ke GitHub & Google Drive, serta membangun Docker Image dan mem-push-nya otomatis ke Docker Hub.

---
## Fitur Utama

- **Training Otomatis Model Random Forest** dengan MLflow
- **Logging otomatis** parameter, metrics, & model dengan MLflow Tracking
- **Backup Artefak Training** ke GitHub & Google Drive
- **Build Docker Image Otomatis** dari model terlatih menggunakan `mlflow models build-docker`
- **Push Docker Image Otomatis** ke Docker Hub

---
## Struktur Proyek

```
Workflow-CI/
├── .github/
│   ├── workflows/
│   │   └── ci_model_training.yml      # Workflow GitHub Actions utama
│
├── MLProject/
│   ├── modelling.py                       # Script training model
│   ├── conda.yaml                         # Spesifikasi environment Conda
│   ├── MLProject                          # Konfigurasi MLflow Project
│   ├── preprocessed_data_auto_/           # Folder dataset hasil preprocessing
│   ├── upload_to_gdrive.py                # Script upload otomatis ke Google Drive
│   ├── docker-hub-link.txt                # Link Docker Hub
    └── gdrive-link.txt                    # Link Gdrive
│
├── Mlflow-Artifact/                       # Folder artefak MLflow hasil training
├── README.md                             
```

---
## Cara Kerja Workflow

### Trigger Otomatis

- Push ke branch `main`
- Pull Request ke branch `main`
- Manual via GitHub Actions (workflow\_dispatch)

### Tahapan yang Dilakukan Workflow
Workflow otomatis di .github/workflows/ci_model_training.yml akan:
1. Setup Conda Environment dari `conda.yaml`
2. Training Model MLflow secara otomatis
3. Backup Artefak MLflow ke folder timestamped
4. Commit & Push Artefak ke Repository GitHub
5. Upload Artefak ke Google Drive
6. Build Docker Image dari model MLflow
7. Push Docker Image ke Docker Hub

### Tahapan Training Otomatis
Script modelling.py secara otomatis menjalankan tahapan berikut:
1. Load Data: Membaca dataset hasil preprocessing.
2. Train Model: Training Random Forest Classifier.
3. MLflow Autologging: Logging otomatis semua parameter & metrics.
4. Custom Metrics Logging: Logging tambahan seperti log loss & training time.
5. Save Model: Menyimpan model ke artefak MLflow.

### Docker Image (Hasil Deployment)
Image otomatis dibuat & dipush ke Docker Hub:
```
docker pull mrickyr/pipe-condition-model
```

---
## Tools & Library
- Python 3.12.7
- MLflow 2.19.0
- Scikit-Learn 1.7.0
- Google Drive API (untuk upload artefak)
- Docker (untuk build & push image)

---
## Notes on This Project
- Pastikan Secrets berikut sudah diset di GitHub:
  - `GDRIVE_CREDENTIALS`
  - `GDRIVE_FOLDER_ID`
  - `DOCKER_HUB_USERNAME`
  - `DOCKER_HUB_ACCESS_TOKEN`
- Dockerfile TIDAK perlu manual, karena otomatis dibuat oleh MLflow.
