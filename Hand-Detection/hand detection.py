import cv2
import mediapipe as mp
import math
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


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


# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                wrist = landmarks[0]

                # Draw hand connections
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(250, 44, 90), thickness=2, circle_radius=1)
                )

                # Check each finger
                finger_info = [
                    (4, "Thumb", 2), (8, "Index", 5),
                    (12, "Middle", 9), (16, "Ring", 13),
                    (20, "Pinky", 17)
                ]

                extended_count = 0
                for tip_id, name, mcp_id in finger_info:
                    mcp = landmarks[mcp_id]
                    if is_finger_extended(landmarks, tip_id, wrist, mcp):
                        extended_count += 1
                        cx, cy = int(landmarks[tip_id].x * w), int(landmarks[tip_id].y * h)
                        cv2.circle(image, (cx, cy), 10, (0, 255, 0), -1)
                        cv2.putText(image, f"{name}", (cx - 20, cy - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display extended fingers count
                cv2.putText(image, f"Extended: {extended_count}/5", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Orientation-Invariant Hand Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()