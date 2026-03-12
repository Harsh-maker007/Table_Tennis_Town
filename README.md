# Table Tennis Rally Analyzer

Streamlit app that tracks table tennis rallies, ball path, and basic player stats using OpenCV + YOLOv8.

## Deploy on Streamlit Community Cloud
1. Go to Streamlit Community Cloud.
2. Click **New app**.
3. Select repository `Harsh-maker007/Table_Tennis_Town`.
4. Branch: `master`
5. Main file path: `app.py`
6. Click **Deploy**.

## Run locally
```bash
streamlit run app.py
```

## Notes
- YOLOv8 weights (e.g., `yolov8n.pt`) will download on first run.
- Face model is auto-downloaded if missing.
