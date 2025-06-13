# 🏃‍♂️ Hand-Controlled Subway Surfers

Control **Subway Surfers** (or any PC game) using just your **hand gestures** via your webcam!

This project uses **MediaPipe** for real-time hand tracking, an on-screen **joystick overlay**, and **keyboard emulation** via `pyautogui` to simulate in-game controls.

---

## 🎮 Features

- ✅ Real-time hand tracking with MediaPipe
- 👉 Swipe-based direction control (left/right/up/down)
- ✊ Double-fist gesture triggers **Skateboard**
- ⭕ On-screen virtual joystick with directional hints
- ⏸ Game auto-pauses when hand leaves control circle
- ▶️ Resumes automatically when hand returns
- 🖥️ Always-on-top joystick window (for Windows)

---

## 🧰 Technologies Used

| Tool         | Purpose                          |
|--------------|-----------------------------------|
| 🐍 Python     | Programming language              |
| 📷 OpenCV     | Webcam & UI rendering             |
| ✋ MediaPipe  | Hand detection & gesture tracking |
| ⌨️ PyAutoGUI  | Keyboard input simulation          |
| 🪟 ctypes     | Always-on-top window (Windows)    |

---

## 🕹️ Controls

| Action       | Gesture Description                     |
|--------------|------------------------------------------|
| Move Left    | 🖐️ Palm swipes left or tilts left         |
| Move Right   | 🖐️ Palm swipes right or tilts right       |
| Jump         | ✋ Hand raised upward                    |
| Duck         | ✋ Hand moved downward                   |
| Pause        | ❌ Remove hand from frame                |
| Resume       | ✋ Show open palm with 3+ fingers        |
| Restart      | 🔁 Press `R`                             |
| Exit         | ❌ Press `ESC`                           |
