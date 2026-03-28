import cv2
import mediapipe as mp
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("カメラから映像を取得できませんでした。")
        break

    # Convert the image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(img_rgb)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the result
    cv2.imshow('Hand Tracking', img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()