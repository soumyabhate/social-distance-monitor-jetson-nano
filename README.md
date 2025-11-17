
# Social Distance Monitor (Jetson poseNet)

Real-time **social-distance visualizer** built on NVIDIA Jetson and `jetson-inference`’s poseNet.

The script:
- Captures live video from a V4L2 camera (`/dev/video0` → fallback to `/dev/video1`)
- Detects people using poseNet (`resnet18-body`)
- Computes the **Euclidean distance** between people (with hip → shoulder fallback)
- Draws **green lines** for safe distances and **red lines** when people are too close
- Logs counts of people and violations to `room.log`

---

## Features

- Runs entirely on Jetson (Nano / Xavier / Orin)
- Uses poseNet `resnet18-body`
- Hip keypoints + Shoulder fallback for center calculation
- Euclidean distance for accurate measurement
- Configurable threshold
- Auto-generated `room.log`

---

# Jetson Setup & Running Instructions

These are the **exact steps used on the Jetson**, without any Git cloning instructions.

---

## 1. Confirm jetson-inference is installed

```bash
python3 -c "import jetson_inference, jetson_utils; print('OK')"
```

If this prints `OK`, the environment is good.

---

## 2. Run the script (IMPORTANT: Do NOT run as root)

Jetson prevents root from opening X11/OpenGL windows.  
Running as root causes:

```
failed to open X11 server connection
monitor: stopped after 0 frames
```

So always run as normal user:

```bash
export DISPLAY=:0
python3 social_distance.py
```

This opens the camera window and draws lines between detected people.

---

## 3. About the Log File (`room.log`)

The script **automatically creates** `room.log`.

Example entry:

```
2025-11-17 12:34:56,poses=5,people=3,viol=2
```

- **poses** = persons detected by poseNet  
- **people** = valid centers after hip/shoulder fallback  
- **viol** = violation count (distance < threshold)  

No need to manually create or upload the log file.

---

# Why Shoulder Keypoints (5 & 6) Were Added

Originally the script used:

- Left hip (ID 11)  
- Right hip (ID 12)  

These determine the body center.

### ❌ Issue in real scenes
Many camera angles **do not capture hips**, such as:

- Top-down / CCTV angles  
- Crowd scenes  
- Mid-body cropping (hips out of frame)  

poseNet often fails to detect hips in these conditions.

Resulting in:

```
people = 0
viol = 0
```

even when many people are visible.

---

# ✔️ Shoulder Fallback (Fix)

If hips are missing, we now automatically use:

- Left shoulder (ID 5)  
- Right shoulder (ID 6)  

Reasons why shoulders work better:

- Shoulders are almost **always visible**
- poseNet detects shoulders more reliably
- Helps in crowded rooms
- Works with chest-level or overhead cameras

---

# 🔥 Final Result

Thanks to hip → shoulder fallback:

- Detection is stable  
- Center points exist even when hips are missing  
- `room.log` correctly reflects people count  
- Violations trigger without failure  
- The system works in real-world environments  

---

# How It Works (Flow)

1. poseNet detects people  
2. Attempt using hip keypoints  
3. If hips missing → fallback to shoulders  
4. Compute person center  
5. Compute Euclidean distance between every pair  
6. Draw lines and count violations  
7. Log everything to `room.log`  

---

# Logging Format

```
YYYY-MM-DD HH:MM:SS,poses=X,people=Y,viol=Z
```

---

# Author

Replace with your actual details:

- **Your Name**  
- GitHub: https://github.com/your-username
