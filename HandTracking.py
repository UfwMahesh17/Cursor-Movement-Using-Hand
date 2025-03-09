import cv2
import mediapipe as mp
import pyautogui

Cam = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
screen_w, screen_h = pyautogui.size()

clicking = False  # Track if dragging is active

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

while True:
    _, frame = Cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hands.process(rgb_frame)
    frame_h, frame_w, _ = frame.shape

    if output.multi_hand_landmarks:
        for hand_landmarks in output.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = hand_landmarks.landmark
            index_tip = (int(landmarks[8].x * frame_w), int(landmarks[8].y * frame_h))
            thumb_tip = (int(landmarks[4].x * frame_w), int(landmarks[4].y * frame_h))
            middle_tip = (int(landmarks[12].x * frame_w), int(landmarks[12].y * frame_h))

            # Move Mouse Pointer
            screen_x = screen_w / frame_w * index_tip[0]
            screen_y = screen_h / frame_h * index_tip[1]
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)  # Smooth movement

            # Draw landmarks
            cv2.circle(frame, index_tip, 8, (0, 255, 0), -1)  # Index Finger
            cv2.circle(frame, thumb_tip, 8, (0, 0, 255), -1)  # Thumb
            cv2.circle(frame, middle_tip, 8, (255, 0, 0), -1)  # Middle Finger

            # Click Detection
            pinch_dist = distance(index_tip, thumb_tip)
            right_click_dist = distance(middle_tip, thumb_tip)

            if pinch_dist < 70:  # Lowered threshold for better sensitivity
                if not clicking:
                    print("Left Click!")
                    pyautogui.click()
                    clicking = True  # Prevent rapid-fire clicking
            elif right_click_dist < 70:
                if not clicking:
                    print("Right Click!")
                    pyautogui.rightClick()
                    clicking = True
            else:
                clicking = False  # Reset click state when fingers separate

    cv2.imshow("Hand Controlled Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

Cam.release()
cv2.destroyAllWindows()
