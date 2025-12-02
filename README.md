# Real-time Road Accident Detection
The project aims to develop a computer vision system capable of detecting road accidents in real-time video
streams. Using temporal recurrent networks and anomaly detection techniques, the system will analyze traffic
footage and identify abnormal events such as collisions or sudden stops.
# Our goal
The final goal is to build a prototype that can automatically monitor traffic cameras or dashcam footage and
issue alerts when an accident occurs. This reduces the reaction time of emergency services, minimizes human
supervision effort, and helps improve road safety.
# Importance
Road accidents cause many injuries and deaths worldwide. Fast detection and reporting can save lives by
bringing quicker medical help. Automated monitoring also helps traffic management and lowers costs from delays.
# How to launch

## Quick Start
If you already have datasets downloaded, just run:
```bash
./train.sh
```
This will:
- Create virtual environment
- Install dependencies
- Train the model (~15-30 min)
- Copy model to the right location
- Restart Docker services

## Option 1: Automatic Setup (First-time users)
Run the automatic setup script:
```bash
./scripts/first_run.sh
```

## Option 2: Manual Training
For manual training you have to: 
- run all cells in `src/data/data_download.ipynb`
- write `python -m src.models.model_train.baseline_vgg` in root directory

## Option 3: Local Inference (without cloud service)
For local inference launch you have to:
- write `python3 -m src.realtime.run_realtime --camera 0` for realtime camera inference
- write `python3 -m src.realtime.run_realtime --video your.mp4` for video inference

## Option 4: Cloud Service Deployment
For cloud service deployment (with WebSocket + REST API + Loki + Grafana):
- write `docker-compose up -d` to start all services (API server, Loki, Promtail, Grafana)
- connect client with `python src/cloud/client.py --camera 0` for video streaming
- access Grafana dashboard at http://localhost:3000 (admin/admin)
- see detailed documentation in `src/cloud/README.md`

**Note**: You need a trained model at `dict_models/vgg16_baseline.pth` for the cloud service to work. Use Option 1 or 2 to get it.
# Contributors
Name Surname|Mail|Position
-|-|-
Danis Sharafiev|d.sharafiev@innopolis.university|ML Engineer
Anton Korotkov|a.korotkov@innopolis.university|Data Scientiest
Alex Kachmazov|a.kachmazov@innopolis.university|ML Engineer
Nikita Shiyanov|n.shiyanov@innopolis.university|ML Engineer
