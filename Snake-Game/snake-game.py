import cv2
import mediapipe as mp
import math
import numpy as np
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 20
INITIAL_SPEED = 10
FOOD_SIZES = {
    'small': {'size': 10, 'points': 2, 'color': (0, 255, 0)},  # Green
    'medium': {'size': 15, 'points': 5, 'color': (0, 165, 255)},  # Orange
    'large': {'size': 20, 'points': 10, 'color': (0, 0, 255)}  # Red
}
TARGET_SCORE = 100
CIRCLE_RADIUS = 230  # Smaller radius for the control circle
CIRCLE_CENTER = (SCREEN_WIDTH - 500, 300)  # Top right position


def calculate_3d_angle(a, b, c):
    """Calculate 3D angle between three points in space"""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1, 1))
    return np.degrees(angle)


def is_hand_open(landmarks):
    """Check if hand is open by examining multiple fingers"""
    wrist = landmarks[0]
    fingers_extended = 0

    # Check each finger (except thumb)
    for finger_tip_id, mcp_id in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        mcp = landmarks[mcp_id]
        pip = landmarks[finger_tip_id - 1]
        dip = landmarks[finger_tip_id - 2]
        tip = landmarks[finger_tip_id]

        # Calculate angles
        v1 = np.array([mcp.x - wrist.x, mcp.y - wrist.y, mcp.z - wrist.z])
        v2 = np.array([pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z])
        v3 = np.array([tip.x - pip.x, tip.y - pip.y, tip.z - pip.z])

        v1 = v1 / (np.linalg.norm(v1) + 1e-7)
        v2 = v2 / (np.linalg.norm(v2) + 1e-7)
        v3 = v3 / (np.linalg.norm(v3) + 1e-7)

        angle1 = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))
        angle2 = np.degrees(np.arccos(np.clip(np.dot(v2, v3), -1, 1)))

        if angle1 < 30 and angle2 < 30:
            fingers_extended += 1

    # Consider hand open if at least 3 fingers are extended
    return fingers_extended >= 3


def get_palm_position(landmarks):
    """Calculate palm center position"""
    # Use wrist (0), mcp joints (1,5,9,13,17) to find palm center
    x = (landmarks[0].x + landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 5
    y = (landmarks[0].y + landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 5
    return (x, y)


class SnakeGame:
    def __init__(self):
        self.snake = [(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)]
        self.direction = (1, 0)  # Initial direction: right
        self.score = 0
        self.speed = INITIAL_SPEED
        self.food = self.generate_food()
        self.game_over = False
        self.paused = False
        self.last_direction_change = time.time()
        self.min_direction_change_interval = 0.1  # seconds

    def generate_food(self):
        food_type = random.choice(list(FOOD_SIZES.keys()))
        size_info = FOOD_SIZES[food_type]

        while True:
            x = random.randint(size_info['size'], SCREEN_WIDTH - size_info['size'])
            y = random.randint(size_info['size'], SCREEN_HEIGHT - size_info['size'])

            # Make sure food doesn't spawn on snake
            valid_position = True
            for segment in self.snake:
                if math.sqrt((x - segment[0]) ** 2 + (y - segment[1]) ** 2) < size_info['size'] + GRID_SIZE:
                    valid_position = False
                    break

            if valid_position:
                return {'x': x, 'y': y, 'type': food_type, 'size': size_info['size'],
                        'points': size_info['points'], 'color': size_info['color']}

    def update(self, palm_pos):
        if self.game_over or self.paused or not palm_pos:
            return

        # Calculate new direction based on palm position relative to circle
        palm_x, palm_y = palm_pos
        dx = palm_x - CIRCLE_CENTER[0]
        dy = CIRCLE_CENTER[1] - palm_y  # Invert dy
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # Only change direction if palm is outside the deadzone (20% of radius)
        if distance > CIRCLE_RADIUS * 0.2:
            # Normalize the direction vector
            if distance > 0:
                dx /= distance
                dy /= distance

            # Determine primary direction based on angle
            angle = math.degrees(math.atan2(dy, dx))

            # Only allow direction changes at certain intervals to prevent erratic movement
            current_time = time.time()
            if current_time - self.last_direction_change >= self.min_direction_change_interval:
                # Up (90° ± 45°)
                if 45 <= angle < 135:
                    new_direction = (0, -1)
                # Right (0° ± 45° or -0° ± 45°)
                elif -45 <= angle < 45:
                    new_direction = (1, 0)
                # Down (-90° ± 45°)
                elif -135 <= angle < -45:
                    new_direction = (0, 1)
                # Left (180° ± 45°)
                else:
                    new_direction = (-1, 0)

                # Prevent 180-degree turns
                if (new_direction[0] * self.direction[0] + new_direction[1] * self.direction[1]) >= 0:
                    self.direction = new_direction
                    self.last_direction_change = current_time

        # Move snake
        head_x, head_y = self.snake[0]
        head_x = (head_x + self.direction[0] * GRID_SIZE) % SCREEN_WIDTH
        head_y = (head_y + self.direction[1] * GRID_SIZE) % SCREEN_HEIGHT
        new_head = (head_x, head_y)

        # Check for collisions with self
        if new_head in self.snake[1:]:
            self.game_over = True
            return

        self.snake.insert(0, new_head)

        # Check for food collision
        food = self.food
        distance = math.sqrt((head_x - food['x']) ** 2 + (head_y - food['y']) ** 2)

        if distance < food['size'] + GRID_SIZE // 2:
            self.score += food['points']
            # Increase speed every 20 points
            self.speed = INITIAL_SPEED + (self.score // 20) * 3 # <--- MODIFIED LINE

            # Generate new food
            self.food = self.generate_food()
        else:
            # Only remove tail if no food was eaten
            self.snake.pop()

    def draw(self, frame):
        # Draw game border
        cv2.rectangle(frame, (0, 0), (SCREEN_WIDTH, SCREEN_HEIGHT), (255, 255, 255), 2)

        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = (0, 255, 0) if i == 0 else (0, 200, 0)  # Head is brighter green
            cv2.circle(frame, (x, y), GRID_SIZE // 2, color, -1)

        # Draw food
        food = self.food
        cv2.circle(frame, (food['x'], food['y']), food['size'], food['color'], -1)

        # Draw score
        cv2.putText(frame, f"Score: {self.score}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw speed
        cv2.putText(frame, f"Speed: {self.speed}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.score >= TARGET_SCORE:
            cv2.putText(frame, "YOU WIN!", (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"Score: {self.score}", (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'R' to Restart", (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.game_over = True  # Keep this line to stop the game
            # Draw game over message ONLY if game_over is true AND score is less than target (meaning you lost)
        elif self.game_over and self.score < TARGET_SCORE:  # <--- CHANGED THIS LINE
            cv2.putText(frame, "GAME OVER!", (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, f"Final Score: {self.score}", (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'R' to Restart", (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw pause message
        if self.paused:
            cv2.putText(frame, "PAUSED", (SCREEN_WIDTH // 2 - 60, SCREEN_HEIGHT // 2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
            cv2.circle(frame, (SCREEN_WIDTH // 2 - 30, SCREEN_HEIGHT // 2 + 30), 20, (255, 255, 0), -1)
            cv2.circle(frame, (SCREEN_WIDTH // 2 + 30, SCREEN_HEIGHT // 2 + 30), 20, (255, 255, 0), -1)


def main():
    cap = cv2.VideoCapture(0)

    # Set camera resolution to match game aspect ratio
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

    game = SnakeGame()
    last_update_time = time.time()
    pause_start_time = 0
    countdown_start = 0
    countdown_value = 0

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            # Resize camera frame to match game height while maintaining aspect ratio
            scale_factor = SCREEN_HEIGHT / frame.shape[0]
            new_width = int(frame.shape[1] * scale_factor)
            resized_frame = cv2.resize(frame, (new_width, SCREEN_HEIGHT))

            # Create game frame
            game_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

            # Process hand landmarks
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            palm_pos = None
            hand_closed = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark

                    # Check if hand is open or closed
                    hand_closed = not is_hand_open(landmarks)

                    # Get palm position
                    palm_x, palm_y = get_palm_position(landmarks)
                    palm_x = int(palm_x * new_width)
                    palm_y = int(palm_y * SCREEN_HEIGHT)
                    palm_pos = (palm_x, palm_y)

                    # Draw hand landmarks on camera frame
                    mp_drawing.draw_landmarks(
                        resized_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(250, 44, 90), thickness=2, circle_radius=1)
                    )

            # Draw control circle on camera feed
            cv2.circle(resized_frame, CIRCLE_CENTER, CIRCLE_RADIUS, (0, 255, 255), 2)

            # Draw directional indicators
            cv2.line(resized_frame,
                     (CIRCLE_CENTER[0], CIRCLE_CENTER[1] - int(CIRCLE_RADIUS * 0.7)),
                     (CIRCLE_CENTER[0], CIRCLE_CENTER[1] - int(CIRCLE_RADIUS * 0.3)),
                     (0, 255, 0), 2)  # Up
            cv2.line(resized_frame,
                     (CIRCLE_CENTER[0] + int(CIRCLE_RADIUS * 0.7), CIRCLE_CENTER[1]),
                     (CIRCLE_CENTER[0] + int(CIRCLE_RADIUS * 0.3), CIRCLE_CENTER[1]),
                     (0, 255, 0), 2)  # Right
            cv2.line(resized_frame,
                     (CIRCLE_CENTER[0], CIRCLE_CENTER[1] + int(CIRCLE_RADIUS * 0.7)),
                     (CIRCLE_CENTER[0], CIRCLE_CENTER[1] + int(CIRCLE_RADIUS * 0.3)),
                     (0, 255, 0), 2)  # Down
            cv2.line(resized_frame,
                     (CIRCLE_CENTER[0] - int(CIRCLE_RADIUS * 0.7), CIRCLE_CENTER[1]),
                     (CIRCLE_CENTER[0] - int(CIRCLE_RADIUS * 0.3), CIRCLE_CENTER[1]),
                     (0, 255, 0), 2)  # Left

            # Handle pause/resume based on hand state
            if hand_closed and not game.paused and not game.game_over:
                game.paused = True
                pause_start_time = time.time()
            elif not hand_closed and game.paused and not game.game_over:
                if countdown_value == 0:
                    countdown_start = time.time()
                    countdown_value = 3

            # Handle countdown when resuming
            if countdown_value > 0:
                elapsed = time.time() - countdown_start
                remaining = countdown_value - int(elapsed)

                if remaining > 0:
                    cv2.putText(game_frame, str(remaining),
                                (SCREEN_WIDTH // 2 - 10, SCREEN_HEIGHT // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                else:
                    game.paused = False
                    countdown_value = 0

            # Game update logic
            current_time = time.time()
            if not game.game_over and not game.paused and countdown_value == 0:
                if current_time - last_update_time > 1.0 / game.speed:
                    game.update(palm_pos)
                    last_update_time = current_time

            # Draw game
            game.draw(game_frame)

            # Combine camera feed and game (side by side)
            combined = np.hstack((resized_frame, game_frame))

            cv2.imshow('Hand-Controlled Snake Game', combined)

            # Handle keyboard input
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC to exit
                break
            elif key == ord('r'):  # R to restart
                game = SnakeGame()
                last_update_time = time.time()
                pause_start_time = 0
                countdown_start = 0
                countdown_value = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
