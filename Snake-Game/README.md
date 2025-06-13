# 🐍 Hand-Controlled Snake Game (CV_snake_game)

Play the classic Snake game — **no keyboard, no touch** — just your **hand gestures**!  
This project uses **MediaPipe** and **OpenCV** to track your hand via webcam and control the snake in real time.

---

## ✨ Features

- ✋ Control snake direction with hand movement
- 🟢 Palm inside control circle → move (based on tilt)
- 🚫 Palm outside → game auto-pauses
- ⏳ Resume with a countdown once hand returns
- 🧠 Visual on-screen joystick with arrow hints
- 🍎 Multiple food types with different scores & sizes
- 🏆 Win the game when you score 100 points
- 🔁 Restart (R), Exit (ESC)

---

## 🧠 Technologies Used

| Tool       | Purpose                            |
|------------|-------------------------------------|
| 🐍 Python   | Programming language                |
| 🎥 OpenCV   | Webcam feed & rendering             |
| ✋ MediaPipe| Hand detection & landmark tracking  |
| ➗ NumPy    | Coordinate and vector math          |

---

## ✋ Controls
Palm inside circle → Snake moves in the direction of palm tilt

Palm out of circle → Game pauses

Open hand (≥ 3 fingers) → Resume game

Press R → Restart game

Press ESC → Exit game

## 📷 Visual Guide
A circle on the screen represents the control zone.

Arrows inside the circle show movement directions.

Hand landmarks are drawn to visualize tracking.

## 🏆 Objective
Reach a score of 100 by eating food items:

🟢 Small (2 points)

🟠 Medium (5 points)

🔴 Large (10 points)

## ⚠️ Notes
Ensure good lighting for accurate hand detection.

Performance may vary depending on webcam quality and processing power.
