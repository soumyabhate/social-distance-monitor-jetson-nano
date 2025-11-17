# Social Distance Monitor (Jetson poseNet)

Real‑time **social‑distance visualizer** built on NVIDIA Jetson and `jetson-inference`’s poseNet.

The script:
- Captures live video from a V4L2 camera (`/dev/video0` → fallback to `/dev/video1`)
- Detects people using poseNet (`resnet18-body`)
- Computes the **Euclidean distance** between people (based on hip keypoints)
- Draws **green lines** for safe distances and **red lines** when people are too close
- Logs counts of people and violations to `room.log`

---

## Demo (Concept)

> ✔️ Live camera feed  
> ✔️ Colored lines between detected people  
> ✔️ `ALERT` status when violations are detected  

(You can add a GIF or screenshot here later.)

---

## Features

- ✅ Runs **entirely on-device** on NVIDIA Jetson
- ✅ Automatic camera selection (`/dev/video0` → `/dev/video1`)
- ✅ Uses poseNet `resnet18-body` model from `jetson-inference`
- ✅ **Euclidean distance** calculation between people
- ✅ Configurable pixel threshold (`THRESHOLD_PX`) for safe distance
- ✅ Simple logging to `room.log` (timestamp, people count, violations)
- ✅ Minimal, single‑file script (`social_distance.py`)

---

## Project Structure

```text
.
├── social_distance.py      # Main script (poseNet social-distance demo)
├── room.log                # Auto-generated log file (created at runtime)
└── README.md               # This file
```

> Rename the script to `social_distance.py` if it currently has a different name before pushing to GitHub.

---

## Requirements

### Hardware

- NVIDIA Jetson device (Nano / Xavier / Orin, etc.)
- USB or CSI camera compatible with V4L2
- Display connected (HDMI) or remote display forwarding (e.g., VNC)

### Software

- JetPack with CUDA and cuDNN properly installed
- [`jetson-inference`](https://github.com/dusty-nv/jetson-inference) library installed with Python bindings
- Python 3 (typically `python3` on Jetson)
- `jetson.utils` and `jetson_inference` Python modules available

---

## Installation

1. **Clone this repository**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

2. **Make sure `jetson-inference` is installed**

Follow the official instructions for building and installing `jetson-inference` with Python bindings on Jetson.

(Short version – just a reminder, not the full guide):

```bash
# Example (do this outside your repo, once per device)
cd ~
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

3. **Verify Python modules**

From Python on your Jetson, verify:

```bash
python3 -c "import jetson_inference, jetson_utils; print('OK')"
```

If you see `OK` with no errors, you’re good.

---

## Usage

From inside your project folder:

```bash
python3 social_distance.py
```

By default, the script will:

- Open `/dev/video0`  
- If that fails, try `/dev/video1`  
- Open a display window (`display://0`)  
- Start processing frames in real time

To **stop** the script:

- Close the display window, or  
- Use `Ctrl + C` in the terminal

---

## Configuration

### Social-distance threshold

At the top of the script, you’ll see:

```python
THRESHOLD_PX = 180  # pixel threshold for too-close
```

- `THRESHOLD_PX` is the **pixel distance** threshold between two people’s hip centers.
- If the Euclidean distance between two people is **less than** this value, the line between them is **red** (violation).
- Otherwise, it’s **green** (safe).

You can tune this value according to:

- Your camera’s resolution
- Field of view
- Physical layout of the room

> Example:  
> - Closer camera / smaller scene → use **larger** threshold (e.g., 200–250)  
> - Wider shot / more zoomed-out → use **smaller** threshold (e.g., 130–180)

---

## How It Works (Internals)

The core logic is in three steps:

### 1. Pose detection

The script loads poseNet:

```python
from jetson_inference import poseNet
net = poseNet("resnet18-body", threshold=0.15)
```

For each frame:

```python
poses = net.Process(img)
```

Each `pose` is a detected person with multiple keypoints.

---

### 2. Compute person centers (using hips)

For every detected person:

```python
L = kpt_xy(p, 11)   # left_hip
R = kpt_xy(p, 12)   # right_hip
if L and R:
    centers.append(((L[0] + R[0]) * 0.5, (L[1] + R[1]) * 0.5))
```

- It uses **left hip (11)** and **right hip (12)** keypoints.
- Takes the average of their `(x, y)` coordinates as the person’s **center**.

---

### 3. Euclidean distance between all pairs

For every pair of detected people:

```python
dx = ax - bx
dy = ay - by
d = (dx**2 + dy**2) ** 0.5  # Euclidean distance
bad = d < THRESHOLD_PX
color = (255, 0, 0, 255) if bad else (0, 255, 0, 255)
jetson.utils.cudaDrawLine(img, (int(ax), int(ay)), (int(bx), int(by)), color, 3)
```

- `d` is the **Euclidean distance** between two centers.
- If `d < THRESHOLD_PX`, the line is **red** and it counts as a **violation**.
- If `d >= THRESHOLD_PX`, the line is **green**.

It also draws small circles on each person center:

```python
jetson.utils.cudaDrawCircle(img, (int(cx), int(cy)), 6, (0, 200, 255, 255))
```

---

## Logging

The script writes to a log file:

```text
room.log
```

Each line looks like:

```text
2025-11-17 10:32:45,people=3,viol=1
```

Where:

- `people` = number of detected people (with valid hip keypoints)
- `viol`   = number of violating pairs (too close)

This can be used later for:

- Simple analytics (how crowded the room was)
- Estimating compliance over time

---

## Troubleshooting

### 1. No camera / black window

- Check that your camera is connected.
- Run:

```bash
ls /dev/video*
```

You should see `/dev/video0` (and maybe `/dev/video1`).
If your camera is on a different index, update `open_camera()` in the script.

### 2. `ModuleNotFoundError: No module named 'jetson_inference'`

- Make sure you completed the `jetson-inference` build and install steps.
- Confirm:

```bash
python3 -c "import jetson_inference, jetson_utils"
```

### 3. Low FPS / lag

- Reduce camera resolution from the Jetson side.
- Make sure nothing else heavy is running on the device.
- Use a more performance-oriented Jetson (Xavier / Orin) for multiple cameras.

---

## Ideas & Extensions

Some ideas you could add later:

- Save periodic snapshots when violations occur.
- Expose metrics over HTTP (Prometheus / simple REST API).
- Add a simple **web dashboard** that shows live counts.
- Trigger alerts (e.g., sound, LED, or Telegram bot) when violations exceed a threshold.

---

## Contributing

Pull requests and suggestions are welcome!

If you:
- Add support for multiple cameras,
- Improve UI overlays,
- Or integrate this into a larger monitoring system,

feel free to open an issue or send a PR.

---

## License

You can choose your preferred license (MIT, Apache-2.0, etc.).  
Typical setup:

- Add a `LICENSE` file at the root.
- Mention the chosen license here in the README.

---

## Author

- **Your Name**  
- GitHub: [@your-username](https://github.com/your-username)

(Replace these with your actual details before publishing.)

