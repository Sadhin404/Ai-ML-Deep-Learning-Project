# 📦 Model Files & Test Video — Download

The detection program [`vehicle_detection.py`](vehicle_detection.py) needs two files beside it to run: the **YOLOv8 weights** and the **test video**. They are **not stored in this GitHub repository** because GitHub does not allow files larger than 100 MB in a repo — the test video alone is 234 MB.

They are kept in the project's Google Drive folder, exactly like the dataset of the Breast Ultrasound project in this repository.

## 🔗 Download Link

**Google Drive — Traffic Detection → Model folder:**
https://drive.google.com/drive/folders/1Kg839s5E8DNiQAOGJVrLK2Njd572vIlA

(Parent folder with all project files: https://drive.google.com/drive/folders/1vn6tw9FwjThnLv1wfH_GbF1REW7T4GyS)

## 📋 What Is Inside

| File | Size | Needed? |
|---|---|---|
| `traffic.mp4` | 234 MB | ✅ **Yes** — the traffic video the code analyses |
| `yolov8m.pt` | 52 MB | ✅ **Yes** — the model the code loads (Ultralytics can also auto-download it on first run) |
| `yolov8s.pt` | 22.6 MB | Optional — smaller/faster variant, edit the code to use it |
| `yolov8n.pt` | 6.5 MB | Optional — smallest/fastest variant, edit the code to use it |

## ▶️ Where to Put Them

Download `traffic.mp4` and `yolov8m.pt` and place them **in the same folder as `vehicle_detection.py`**, then run:

```bash
pip install ultralytics opencv-python
python vehicle_detection.py
```

The program expects exactly these file names — `traffic.mp4` and `yolov8m.pt` — so do not rename them.
