# 📦 Test Video & YOLOv8m Weight — Download

The detection program [`vehicle_detection.py`](vehicle_detection.py) needs two files beside it to run: the **test video** and the **YOLOv8m weights**.

- `yolov8n.pt` (6.5 MB) and `yolov8s.pt` (22.6 MB) are **already in this folder** — download them straight from here.
- `traffic.mp4` (234 MB) and `yolov8m.pt` (52 MB) are **not stored in this GitHub repository**: GitHub does not allow files larger than 100 MB in a repo, and its web upload limit is 25 MB per file. They are kept in the project's Google Drive folder instead — exactly like the dataset of the Breast Ultrasound project in this repository.

## 🔗 Download Link

**Google Drive — Traffic Detection → Model folder:**
https://drive.google.com/drive/folders/1Kg839s5E8DNiQAOGJVrLK2Njd572vIlA

(Parent folder with all project files: https://drive.google.com/drive/folders/1vn6tw9FwjThnLv1wfH_GbF1REW7T4GyS)

## 📋 The Files

| File | Size | Where |
|---|---|---|
| `traffic.mp4` | 234 MB | ☁️ Google Drive (link above) — **required**, the video the code analyses |
| `yolov8m.pt` | 52 MB | ☁️ Google Drive — the model the code loads (Ultralytics can also auto-download it on first run) |
| `yolov8s.pt` | 22.6 MB | ✅ **In this folder** — smaller/faster variant, edit the code to use it |
| `yolov8n.pt` | 6.5 MB | ✅ **In this folder** — smallest/fastest variant, edit the code to use it |

## ▶️ Where to Put Them

Download `traffic.mp4` and `yolov8m.pt` from Drive and place them **in the same folder as `vehicle_detection.py`**, then run:

```bash
pip install ultralytics opencv-python
python vehicle_detection.py
```

The program expects exactly these file names — `traffic.mp4` and `yolov8m.pt` — so do not rename them.
