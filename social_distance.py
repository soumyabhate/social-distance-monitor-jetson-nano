#!/usr/bin/env python3
# Minimal poseNet social-distance demo (no CLI args)
# - Input: auto-picks /dev/video0 (falls back to /dev/video1)
# - Output: display://0
# - Log: room.log in the same folder

import os, time, math
from jetson_inference import poseNet
import jetson.utils

THRESHOLD_PX = 180  # pixel threshold for too-close

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
LOG_PATH = os.path.join(SCRIPT_DIR, "room.log")

def kpt_xy(pose, idx):
    if idx < len(pose.Keypoints):
        kp = pose.Keypoints[idx]
        if kp.ID >= 0:
            return (kp.x, kp.y)
    return None

def open_camera():
    # try /dev/video0 then /dev/video1
    try_order = ["/dev/video0", "/dev/video1"]
    for uri in try_order:
        try:
            print(f"[INFO] trying camera {uri}")
            cam = jetson.utils.videoSource(uri)
            return cam
        except Exception as e:
            print(f"[WARN] failed opening {uri}: {e}")
    raise RuntimeError("no V4L2 camera available")

def main():
    display = jetson.utils.videoOutput("display://0")
    net = poseNet("resnet18-body", threshold=0.15)
    cam = open_camera()
    print("[INFO] display=display://0  log=", LOG_PATH)

    # open log
    try:
        logf = open(LOG_PATH, "a", buffering=1)
        logf.write(f"--- START {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    except Exception as e:
        logf = None
        print("[WARN] couldn't open log:", e)

    # don't gate the loop on cam.IsStreaming() before first Capture()
    while display.IsStreaming():
        img = cam.Capture()
        if img is None:
            continue

        poses = net.Process(img)

        # centers from hips
        centers = []
        for p in poses:
            L = kpt_xy(p, 11)   # left_hip
            R = kpt_xy(p, 12)   # right_hip
            if L and R:
                centers.append(((L[0] + R[0]) * 0.5, (L[1] + R[1]) * 0.5))
                
            # agar hips missing hain to shoulders ka fallback
            if not (L and R):
                L = kpt_xy(p, 5)    # left_shoulder
                R = kpt_xy(p, 6)    # right_shoulder

            if L and R:
                cx = (L[0] + R[0]) * 0.5
                cy = (L[1] + R[1]) * 0.5
                centers.append((cx, cy))

        # distances + draw
        viol = 0
        for i in range(len(centers)):
            ax, ay = centers[i]
            for j in range(i + 1, len(centers)):
                bx, by = centers[j]
                
                # CHANGED: Euclidean distance instead of math.hypot
                dx = ax - bx
                dy = ay - by
                d = (dx**2 + dy**2) ** 0.5
                
                bad = d < THRESHOLD_PX
                color = (255, 0, 0, 255) if bad else (0, 255, 0, 255)
                jetson.utils.cudaDrawLine(img, (int(ax), int(ay)), (int(bx), int(by)), color, 3)
                # label distance mid-line
                mx, my = int((ax + bx) / 2), int((ay + by) / 2)
                # jetson.utils.cudaPrintOverlay(img, f"{int(d)}", mx - 8, my - 12, (255, 255, 255, 255))
                if bad:
                    viol += 1

        # dots
        for (cx, cy) in centers:
            jetson.utils.cudaDrawCircle(img, (int(cx), int(cy)), 6, (0, 200, 255, 255))

        # HUD
        hud = f"people={len(centers)}  viol={viol}  thr={THRESHOLD_PX}px"
        if viol > 0:
            hud = "ALERT!  " + hud
         # jetson.utils.cudaPrintOverlay(img, hud, 10, 10, (255, 255, 255, 255))

        display.Render(img)
        display.SetStatus("SocialDistance (poseNet)")

        # log
        if logf:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            logf.write(f"{ts},people={len(centers)},viol={viol}\n")

    if logf:
        logf.write(f"--- END {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        logf.close()

if __name__ == "__main__":
    main()
