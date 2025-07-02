# 🧠 Concentration Tracker: Real-Time Focus Detection with OpenCV & MediaPipe

An intelligent **real-time concentration monitoring system** that uses webcam input and facial landmark detection to estimate how focused a person is. Built using **MediaPipe Face Mesh**, the app tracks **eye blinks**, **gaze direction**, and **head pose** to compute a live **concentration score**.

This project is ideal for educational apps, productivity tools, and attention analysis systems.

---

## ✨ Features

- 🧿 **Eye Blink Detection** using Eye Aspect Ratio (EAR)
- 👀 **Gaze Direction Tracking** with iris center landmarks
- 📐 **Head Pose Estimation** using nose alignment
- 📊 **Concentration Score (0–100%)** calculated from multiple signals
- 🟨 **Smooth Score Averaging** via rolling history
- 🔴 **Distraction Alerts** when concentration drops
- 🎛️ **Live Visual Overlay** including:
  - FPS display
  - Smooth concentration bar
  - Blinking alert
  - Distraction counter
  - Status indicator (Active / Distracted)

---

## 🧰 Tech Stack

| Tool           | Purpose                                 |
|----------------|-----------------------------------------|
| **Python**     | Programming language                    |
| **OpenCV**     | Video capture and real-time overlays    |
| **MediaPipe**  | Facial landmark detection (Face Mesh)   |
| **NumPy**      | Signal smoothing & vector computations  |
| **Deque**      | Rolling score average buffer            |

---

