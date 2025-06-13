import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Updated Colors (BGR format)
COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "YELLOW": (0, 255, 255),
    "BLACK": (0, 0, 0),
    "WHITE": (255, 255, 255)  # This will be our eraser
}
current_color = "RED"
color_change_time = None

# Drawing variables
canvas = None
prev_point = None
drawing_mode = False
clear_gesture_counter = 0

cap = cv2.VideoCapture(0)


def calculate_3d_angle(a, b, c):
    """Calculate 3D angle between three points in space"""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1, 1))
    return np.degrees(angle)


def is_finger_extended(landmarks, finger_tip_id, wrist, mcp):
    """Improved finger extension check using relative angles"""
    pip = landmarks[finger_tip_id - 1]
    dip = landmarks[finger_tip_id - 2]
    tip = landmarks[finger_tip_id]

    # Calculate angles in the finger's local coordinate system
    v1 = np.array([mcp.x - wrist.x, mcp.y - wrist.y, mcp.z - wrist.z])  # Wrist to MCP
    v2 = np.array([pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z])  # MCP to PIP
    v3 = np.array([tip.x - pip.x, tip.y - pip.y, tip.z - pip.z])  # PIP to TIP

    # Normalize vectors
    v1 = v1 / (np.linalg.norm(v1) + 1e-7)
    v2 = v2 / (np.linalg.norm(v2) + 1e-7)
    v3 = v3 / (np.linalg.norm(v3) + 1e-7)

    # Calculate angles between segments
    angle1 = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))  # Wrist-MCP-PIP
    angle2 = np.degrees(np.arccos(np.clip(np.dot(v2, v3), -1, 1)))  # MCP-PIP-TIP

    # Dynamic thresholds based on finger type
    if finger_tip_id == 4:  # Thumb (more flexible)
        return angle1 < 30 and angle2 < 30  # More acute angles when extended
    else:  # Other fingers
        return angle1 < 25 and angle2 < 25


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Initialize canvas with white background
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = 255  # White background

    # Draw color selection boxes
    box_size = 60
    margin = 15
    color_boxes = {}
    for i, (color_name, color_val) in enumerate(COLORS.items()):
        x1 = margin + i * (box_size + margin)
        y1 = margin
        x2 = x1 + box_size
        y2 = y1 + box_size
        color_boxes[color_name] = (x1, y1, x2, y2)

        # Draw color box with black border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), color_val, -1)

        # Highlight current selection
        if color_name == current_color:
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 2)

    # Process hand
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark
        wrist = landmarks[0]

        # Get key points
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        middle_tip = landmarks[12]
        middle_mcp = landmarks[9]
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        ring_tip = landmarks[16]
        ring_mcp = landmarks[13]
        pinky_tip = landmarks[20]
        pinky_mcp = landmarks[17]

        # Convert to pixels
        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
        mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

        # Draw pointer (triangle at index finger edge, lowered position)
        pointer_size = 15
        pointer_offset = 25  # Lower the pointer from fingertip
        pointer_base_y = iy + pointer_offset

        # Pointer color matches current drawing color
        pointer_color = COLORS[current_color] if current_color != "WHITE" else (255, 255, 255)

        # Draw triangle pointer (pointing up from lowered position)
        pointer_points = np.array([
            [ix, pointer_base_y - pointer_size],  # Tip
            [ix + pointer_size, pointer_base_y],  # Right base
            [ix - pointer_size, pointer_base_y]  # Left base
        ])
        cv2.drawContours(frame, [pointer_points], 0, pointer_color, -1)
        cv2.drawContours(frame, [pointer_points], 0, (0, 0, 0), 1)  # Black border

        # Finger extension checks
        index_extended = is_finger_extended(landmarks, 8, wrist, index_mcp)
        middle_extended = is_finger_extended(landmarks, 12, wrist, middle_mcp)
        thumb_extended = is_finger_extended(landmarks, 4, wrist, thumb_mcp)
        ring_extended = is_finger_extended(landmarks, 16, wrist, ring_mcp)
        pinky_extended = is_finger_extended(landmarks, 20, wrist, pinky_mcp)

        # Color selection logic
        selecting_color = False
        for color_name, (x1, y1, x2, y2) in color_boxes.items():
            if x1 < ix < x2 and y1 < iy < y2:
                selecting_color = True
                if color_change_time is None:
                    color_change_time = time.time()
                elif time.time() - color_change_time > 1.5:  # Faster color change
                    current_color = color_name
                    color_change_time = None
                # Draw selection timer
                elapsed = time.time() - color_change_time if color_change_time else 0
                cv2.circle(frame, (ix, iy), int(box_size / 2 * min(elapsed / 1.5, 1)),
                           (255, 255, 255), 2)
                break

        if not selecting_color:
            color_change_time = None

        # Clear gesture (open then close hand)
        if all([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
            clear_gesture_counter = 1
        elif clear_gesture_counter == 1 and not any(
                [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
            canvas[:] = 255  # Reset to white
            clear_gesture_counter = 0

        # Drawing logic
        if index_extended and middle_extended and not thumb_extended:
            # Use the pointer position (lowered) for drawing
            draw_point = (ix, pointer_base_y)

            if prev_point:
                # Use white color for "eraser" mode
                line_color = (255, 255, 255) if current_color == "WHITE" else COLORS[current_color]
                line_thickness = 15 if current_color == "WHITE" else 8
                cv2.line(canvas, prev_point, draw_point, line_color, line_thickness)

            # Draw real-time preview
            preview_color = (255, 255, 255) if current_color == "WHITE" else COLORS[current_color]
            cv2.circle(frame, draw_point, 8, preview_color, -1)

            prev_point = draw_point
        else:
            prev_point = None

    # Combine canvas and camera feed
    mask = np.any(canvas != 255, axis=2)  # Detect where drawing exists
    frame[mask] = canvas[mask]
    # Display current mode
    mode_text = "ERASER" if current_color == "WHITE" else f"DRAW: {current_color}"
    text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.rectangle(frame, (w - text_size[0] - 30, h - 60),
                  (w - 10, h - 10), (0, 0, 0), -1)
    cv2.putText(frame, mode_text, (w - text_size[0] - 20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Instructions
    cv2.putText(frame, "Draw: Extend index+middle", (20, h - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, "Clear: Open then close hand", (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, "Eraser: Select WHITE color", (20, h - 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow('Enhanced Air Canvas', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()