# ▶️ How to Run — Complete Beginner Guide

This guide is written for someone who has **just installed VS Code and Python** and has never run a project like this before. Follow the steps in order — do not skip any. In about 10–15 minutes the vehicle detection will be running on your computer.

> **You need:** a Windows PC (steps are written for Windows; macOS/Linux differences are noted), an internet connection, and about 4 GB of free disk space.

---

## Step 1 — Check Python is installed properly

1. Open **Command Prompt** (press `Win + R`, type `cmd`, press Enter).
2. Type this command and press Enter:

```bash
python --version
```

3. You should see something like `Python 3.11.x` or `Python 3.12.x`.

❌ **If you see "'python' is not recognized"** — Python was installed without adding it to PATH. Reinstall Python from [python.org/downloads](https://www.python.org/downloads/) and on the **first screen of the installer, tick the box "Add python.exe to PATH"** before clicking Install. Then close and reopen Command Prompt and try again.

*(macOS/Linux: use `python3 --version` and `pip3` in the commands below.)*

---

## Step 2 — Install the Python extension in VS Code

1. Open **VS Code**.
2. Click the **Extensions** icon on the left sidebar (or press `Ctrl + Shift + X`).
3. Search for **Python** and install the one by **Microsoft**.
4. Restart VS Code once.

---

## Step 3 — Download all the project files into ONE folder

Create a new folder on your computer, for example `D:\VehicleDetection`, and put **all four files** inside it:

| File | Where to get it |
|---|---|
| `vehicle_detection.py` | This folder (GitHub → `Traffic Detection/model/`) |
| `yolov8m.pt` | Google Drive — link in [DOWNLOAD_FILES.md](DOWNLOAD_FILES.md) |
| `traffic.mp4` | Google Drive — link in [DOWNLOAD_FILES.md](DOWNLOAD_FILES.md) |
| *(optional)* `yolov8n.pt`, `yolov8s.pt` | Already in this GitHub folder — extra, smaller models |

⚠️ **Do not rename any file.** The code looks for exactly `yolov8m.pt` and `traffic.mp4`. All files must sit **side by side in the same folder** — no subfolders.

---

## Step 4 — Open the folder in VS Code

1. In VS Code: **File → Open Folder…** and select your `VehicleDetection` folder.
2. You should see all the files listed in the left sidebar.
3. Click on `vehicle_detection.py` to open it.

---

## Step 5 — Open the terminal inside VS Code

Press `` Ctrl + ` `` (Control + backtick) or go to **Terminal → New Terminal**.
A terminal panel opens at the bottom, already inside your folder. **All commands below are typed in this terminal.**

First, check the terminal is in the right folder — type:

```bash
dir
```

You should see `vehicle_detection.py`, `yolov8m.pt` and `traffic.mp4` in the list. If not, you opened the wrong folder — go back to Step 4.

---

## Step 6 — (Recommended) Create a virtual environment

This keeps this project's libraries separate from everything else:

```bash
python -m venv venv
venv\Scripts\activate
```

After activation, the terminal line starts with `(venv)`. Keep this terminal open — if you close it, run the activate command again before running the project.

*(macOS/Linux activation: `source venv/bin/activate`)*

---

## Step 7 — Install the required libraries

With the terminal open in your project folder, run:

```bash
pip install ultralytics opencv-python
```

This installs everything the project needs (Ultralytics YOLO, OpenCV, and PyTorch, which YOLO uses internally).

⏳ **Be patient:** this download is large (PyTorch is ~2–3 GB) and can take **5–15 minutes** depending on your internet speed. Let it finish completely — do not close the terminal while it is running. It is done when you see `Successfully installed ...` and the command prompt returns.

*(Optional: if pip itself is very old, run `python -m pip install --upgrade pip` first.)*

---

## Step 8 — Run the project 🚀

```bash
python vehicle_detection.py
```

**What happens next:**

1. The model loads (a few seconds).
2. A window titled **"YOLO Vehicle Detection"** opens and the traffic video starts playing.
3. Every car, bus, truck and motorbike gets a **green box** with its name and confidence score, e.g. `car 0.95`.
4. To stop: click on the video window and press **Q**. The program prints `Finished` and closes.

🎉 **That's it — you are running AI vehicle detection.**

---

## Step 9 — (Optional) Try a faster, smaller model

The code uses `yolov8m.pt` (medium — best accuracy). Two smaller models are included in this GitHub folder. To use one, open `vehicle_detection.py` and change this line:

```python
model = YOLO("yolov8m.pt")
```

to:

```python
model = YOLO("yolov8n.pt")   # fastest, slightly less accurate
```

or `"yolov8s.pt"`, save the file, and run Step 8 again. On a slow PC, `yolov8n.pt` gives a much smoother video.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Python is not in PATH — reinstall Python and tick **"Add python.exe to PATH"** (Step 1). Or try the `py` command instead of `python`. |
| `No module named 'ultralytics'` or `'cv2'` | The libraries were installed in a different environment. Activate the venv (Step 6), run the Step 7 install again **in the same VS Code terminal**, and make sure VS Code uses the venv: `Ctrl + Shift + P` → **Python: Select Interpreter** → choose the one with `venv` in its path. |
| Program prints `VIdeo loading error` and exits | `traffic.mp4` is missing, renamed, or not in the same folder as the code. Check with `dir` (Step 5) — names must match exactly. |
| `FileNotFoundError: yolov8m.pt` | Same reason — download `yolov8m.pt` from the Drive link in [DOWNLOAD_FILES.md](DOWNLOAD_FILES.md), or delete the file name worry: Ultralytics auto-downloads the weight on first run if you have internet. |
| Video plays but is slow / laggy | Normal on a CPU-only PC with the medium model. Switch to `yolov8n.pt` (Step 9) for smooth playback. |
| A window opens but the video already ended | The code processes the first 5,400 frames of the video. Just run it again (Step 8). |
| pip install fails with network errors | Check your internet connection and run the command again — pip resumes where it can. A VPN or campus network firewall can also block it; try another network. |

---

**Still stuck?** Re-read the step where it failed — in almost every case the cause is one of three things: Python not in PATH (Step 1), libraries installed in the wrong environment (Step 7 + Troubleshooting row 2), or the files not being together in one folder (Step 3).
