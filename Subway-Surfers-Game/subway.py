import cv2
import mediapipe as mp
import pyautogui
import time
import math
import ctypes

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.4)
mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

window_name = "Hand Gesture Control"
cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(window_name, 400, 300)

# Always-on-top window (Windows only)
hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 1 | 2)

# Joystick circle config
joystick_center = (320, 240)  # center of 400x300 screen
joystick_radius = 230

# Movement & pause config
prev_x, prev_y = 0, 0
gesture_cooldown = 0.3
last_gesture_time = 0

# Skateboard config
fist_times = []
fist_threshold = 0.05
skateboard_trigger_gap = 1.0

# Pause state
is_paused = False
pause_toggle_time = 0
pause_cooldown = 1.0

def detect_direction(curr_x, curr_y, prev_x, prev_y):
    dx = curr_x - prev_x
    dy = curr_y - prev_y

    if abs(dx) > 25 and abs(dx) > abs(dy):
        return 'right' if dx > 0 else 'left'
    elif abs(dy) > 25:
        return 'down' if dy > 0 else 'up'
    return None

def is_fist(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
    return dist < fist_threshold

def draw_joystick_overlay(img):
    h, w = img.shape[:2]
    cv2.circle(img, joystick_center, joystick_radius, (255, 255, 255), 2)
    cv2.arrowedLine(img, joystick_center, (joystick_center[0] - 60, joystick_center[1]), (0, 255, 0), 2, tipLength=0.4)
    cv2.arrowedLine(img, joystick_center, (joystick_center[0] + 60, joystick_center[1]), (0, 255, 0), 2, tipLength=0.4)
    cv2.arrowedLine(img, joystick_center, (joystick_center[0], joystick_center[1] - 60), (0, 255, 0), 2, tipLength=0.4)
    cv2.arrowedLine(img, joystick_center, (joystick_center[0], joystick_center[1] + 60), (0, 255, 0), 2, tipLength=0.4)
    cv2.putText(img, "Fist x2 = Skateboard", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def is_inside_circle(x, y, center, radius):
    return math.hypot(x - center[0], y - center[1]) < radius

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    h, w, _ = frame.shape

    draw_joystick_overlay(frame)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            cx = int(handLms.landmark[0].x * w)
            cy = int(handLms.landmark[0].y * h)

            # ⏸ Pause if hand moves out of joystick
            in_circle = is_inside_circle(cx, cy, joystick_center, joystick_radius)
            current_time = time.time()

            if not in_circle and not is_paused and (current_time - pause_toggle_time > pause_cooldown):
                pyautogui.press('p')
                print("⏸️ Game Paused")
                is_paused = True
                pause_toggle_time = current_time

            elif in_circle and is_paused and (current_time - pause_toggle_time > pause_cooldown):
                pyautogui.press('p')
                print("▶️ Game Resumed")
                is_paused = False
                pause_toggle_time = current_time

            # ➡️ Direction detection
            if time.time() - last_gesture_time > gesture_cooldown and in_circle:
                gesture = detect_direction(cx, cy, prev_x, prev_y)
                if gesture:
                    pyautogui.press(gesture)
                    print("Detected:", gesture)
                    last_gesture_time = time.time()

            prev_x, prev_y = cx, cy

            # 🛹 Skateboard trigger (double fist)
            if is_fist(handLms.landmark):
                fist_times.append(current_time)
                fist_times = [t for t in fist_times if current_time - t < 1.5]

                if len(fist_times) >= 2 and current_time - fist_times[-2] < skateboard_trigger_gap:
                    print("🛹 Skateboard Activated")
                    pyautogui.press('space')
                    fist_times.clear()

    # Show frame
    cv2.imshow(window_name, cv2.resize(frame, (400, 300)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
